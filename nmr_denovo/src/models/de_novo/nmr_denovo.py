import typing as T
from abc import ABC

import torch
import torch.nn as nn

import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from torchmetrics.aggregation import MeanMetric

from src.models.nmr_base import NMRBaseModel, Stage
from src.utils import morgan_fp, mol_to_inchi_key, MyopicMCES

import hydra

class NMRDeNovoModel(NMRBaseModel, ABC):

    def __init__(
        self,
        top_ks: T.Iterable[int] = (1, 10),
        myopic_mces_kwargs: T.Optional[T.Mapping] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder_1h = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        self.encoder_13c = hydra.utils.instantiate(self.hparams.encoder, _recursive_=False)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, smiles_tokenizer=self.hparams.tokenizer, _recursive_=False)
        # self.criterion = nn.CrossEntropyLoss()
        self.test_result_path = self.hparams.test_result_path

        self.top_ks = top_ks
        self.myopic_mces = MyopicMCES(**(myopic_mces_kwargs or {}))
        self.mol_pred_kind: T.Literal["smiles", "rdkit"] = "smiles"
        # caches of already computed results to avoid expensive re-computations
        self.mces_cache = dict()
        self.mol_2_morgan_fp = dict()

        self._predictions = {
            'gt_smiles': [],
            'pred_mols': []  
        }

    def forward(self, batch):
        nmr1h_embedding = self.encoder_1h(batch['nmr_1h_spec'])
        nmr13c_embedding = self.encoder_13c(batch['nmr_13c_spec'])
        spec_embedding = torch.cat([nmr1h_embedding, nmr13c_embedding], dim=1)
        loss = self.decoder(spec_embedding, batch)
        return loss, spec_embedding

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:

        # Forward pass
        loss, spec_embedding = self.forward(batch)
        # Compute loss
        # loss = self.criterion(smiles_pred, smiles)
        if stage in self.log_only_loss_at_stages:
            mols_pred = None
        else:
            mols_pred = self.decoder.decode_smiles(spec_embedding, batch)

        return dict(loss=loss, mols_pred=mols_pred)

    def on_batch_end(
        self,
        outputs: T.Any,
        batch: dict,
        batch_idx: int,
        stage: Stage
    ) -> None:
        self.log(
            f"{stage.to_pref()}loss",
            outputs['loss'],
            batch_size=len(batch['smiles']),
            sync_dist=True,
            prog_bar=True,
        )
        if stage in self.log_only_loss_at_stages:
            return
        print("nmrdenovo_eval77 gt", batch["smiles"][0])
        print("nmrdenovo_eval7  pred", outputs["mols_pred"][0])
        metric_vals = self.evaluate_de_novo_step(
            outputs["mols_pred"],  # (bs, k) list of generated rdkit molecules or SMILES strings
            batch["smiles"],  # (bs) list of ground truth SMILES strings
            stage=stage
        )
        if stage.value == Stage.TEST.value:
            self._predictions['gt_smiles'].extend(batch['smiles'])
            self._predictions['pred_mols'].extend(outputs['mols_pred'])
            # self._update_df_test(metric_vals)


    def evaluate_de_novo_step(
        self,
        mols_pred: list[list[T.Optional[Chem.Mol | str]]],
        mol_true: list[str],
        stage: Stage,
    ) -> dict[str, torch.Tensor]:
        """
        # TODO: refactor to compute only for max(k) and then use the result to obtain the rest by
        subsetting.

        Main evaluation method for the models for de novo molecule generation from mass spectra.

        Args:
            mols_pred (list[list[Mol | str]]): (bs, k) list of generated rdkit molecules or SMILES
                strings with possible Nones if no molecule was generated
            mol_true (list[str]): (bs) list of ground-truth SMILES strings
        """
        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # Get SMILES and RDKit molecule objects for all predictions
        if self.mol_pred_kind == "smiles":
            smiles_pred_valid, mols_pred_valid = [], []
            for mols_pred_sample in mols_pred:
                smiles_pred_valid_sample, mols_pred_valid_sample = [], []
                for s in mols_pred_sample:
                    m = Chem.MolFromSmiles(s) if s is not None else None
                    # If SMILES cannot be converted to RDKit molecule, the molecule is set to None
                    smiles_pred_valid_sample.append(s if m is not None else None)
                    mols_pred_valid_sample.append(m)
                smiles_pred_valid.append(smiles_pred_valid_sample)
                mols_pred_valid.append(mols_pred_valid_sample)
            smiles_pred, mols_pred = smiles_pred_valid, mols_pred_valid
        elif self.mol_pred_kind == "rdkit":
            smiles_pred = [
                [Chem.MolToSmiles(m) if m is not None else None for m in ms]
                for ms in mols_pred
            ]
        else:
            raise ValueError(f"Invalid mol_pred_kind: {self.mol_pred_kind}")

        # Auxiliary metric: number of valid molecules
        self._update_metric(
            stage.to_pref() + f"num_valid_mols",
            MeanMetric,
            ([sum([m is not None for m in ms]) for ms in mols_pred],),
            batch_size=len(mols_pred),
        )

        # Get RDKit molecule objects for ground truth
        smile_true = mol_true
        mol_true = [Chem.MolFromSmiles(sm) for sm in mol_true]

        def _get_morgan_fp_with_cache(mol):
            """
            A helper function to retrieve either cached Morgan Fingerprint value, or to compute and cache it
            @param mol: RDKit molecule object
            @return:
            """
            if mol not in self.mol_2_morgan_fp:
                self.mol_2_morgan_fp[mol] = morgan_fp(mol, to_np=False)
            return self.mol_2_morgan_fp[mol]

        # Evaluate top-k metrics
        for top_k in self.top_ks:
            # Get top-k predicted molecules for each ground-truth sample
            smiles_pred_top_k = [smiles_pred_sample[:top_k] for smiles_pred_sample in smiles_pred]
            mols_pred_top_k = [mols_pred_sample[:top_k] for mols_pred_sample in mols_pred]

            # 1. Evaluate minimum common edge subgraph:  
            # Calculate MCES distance between top-k predicted molecules and ground truth and
            # report the minimum distance. The minimum distances for each sample in the batch are
            # averaged across the epoch.
            if stage not in self.no_mces_metrics_at_stages:
                min_mces_dists = []
                mces_thld = 100
                # Iterate over batch
                for preds, true in zip(smiles_pred_top_k, smile_true):
                    # Iterate over top-k predicted molecule samples
                    dists = []
                    for pred in preds:
                        if pred is None:
                            dists.append(mces_thld)
                        else:
                            if (true, pred) not in self.mces_cache:
                                mce_val = self.myopic_mces(true, pred)
                                self.mces_cache[(true, pred)] = mce_val
                            dists.append(self.mces_cache[(true, pred)])
                    min_mces_dists.append(min(min(dists), mces_thld))
                min_mces_dists = torch.tensor(min_mces_dists, device=self.device)

                # Log
                metric_name = stage.to_pref() + f"top_{top_k}_mces_dist"
                self._update_metric(
                    metric_name,
                    MeanMetric,
                    (min_mces_dists,),
                    batch_size=len(min_mces_dists),
                    bootstrap=stage == Stage.TEST
                )
                metric_vals[metric_name] = min_mces_dists

            # 2. Evaluate Tanimoto similarity:
            # Calculate Tanimoto similarity between top-k predicted molecules and ground truth and
            # report the maximum similarity. The maximum similarities for each sample in the batch
            # are averaged across the epoch.
            fps_pred_top_k = [
                [_get_morgan_fp_with_cache(m) if m is not None else None for m in ms]
                for ms in mols_pred_top_k
            ]
            fp_true = [_get_morgan_fp_with_cache(m) for m in mol_true]

            max_tanimoto_sims = []
            # Iterate over batch
            for preds, true in zip(fps_pred_top_k, fp_true):
                # Iterate over top-k predicted molecule samples
                sims = [
                    TanimotoSimilarity(true, pred)
                    if pred is not None else 0
                    for pred in preds
                ]
                max_tanimoto_sims.append(max(sims))
            max_tanimoto_sims = torch.tensor(max_tanimoto_sims, device=self.device)

            # Log
            metric_name = stage.to_pref() + f"top_{top_k}_max_tanimoto_sim"
            self._update_metric(
                metric_name,
                MeanMetric,
                (max_tanimoto_sims,),
                batch_size=len(max_tanimoto_sims),
                bootstrap=stage == Stage.TEST
            )
            metric_vals[metric_name] = max_tanimoto_sims

            # 3. Evaluate exact match (accuracy):
            # Calculate if the ground truth molecule is in the top-k predicted molecules and report
            # the average across the epoch.
            in_top_k = [
                mol_to_inchi_key(true) in [
                    mol_to_inchi_key(pred)
                    if pred is not None else None
                    for pred in preds
                ]
                for true, preds in zip(mol_true, mols_pred_top_k)
            ]
            in_top_k = torch.tensor(in_top_k, device=self.device)

            # Log
            metric_name = stage.to_pref() + f"top_{top_k}_accuracy"
            self._update_metric(
                metric_name,
                MeanMetric,
                (in_top_k,),
                batch_size=len(in_top_k),
                bootstrap=stage == Stage.TEST
            )
            metric_vals[metric_name] = in_top_k

        return metric_vals


    def test_step(
        self,
        batch: dict,
        batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nmr1h_embedding = self.encoder_1h(batch['nmr_1h_spec'])
        nmr13c_embedding = self.encoder_13c(batch['nmr_13c_spec'])
        spec_embedding = torch.cat([nmr1h_embedding, nmr13c_embedding], dim=1)

        mols_pred = self.decoder.decode_smiles(spec_embedding, batch)
        outputs = {
            'mols_pred': mols_pred,
            'loss': -1,
        }
        # Get generated (i.e., predicted) SMILES
        if self.df_test_path is not None:
            self._update_df_test({
                'identifier': batch['identifier'],
                'mols_pred': mols_pred,
            })

        return outputs

    def on_test_epoch_end(self):
        if self.test_result_path is not None:

            gt_list = self._predictions['gt_smiles']          
            pred_lists = self._predictions['pred_mols']       
            
            assert len(gt_list) == len(pred_lists), "len(gt_list) != len(pred_lists)"
            data = {
                "GT_SMILES": gt_list
            }
            for i in range(len(pred_lists[0])):
                data[f"Pred_{i+1}"] = [preds[i] for preds in pred_lists]

            df = pd.DataFrame(data)

            df.to_csv(self.test_result_path, index=False)      

        if self.df_test_path is not None:
            df_test = pd.DataFrame(self.df_test)
            self.df_test_path.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_pickle(self.df_test_path)
