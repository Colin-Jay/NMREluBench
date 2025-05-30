import typing as T
import hydra
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import src.utils as utils
from pathlib import Path
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from src.data.nmr_datasets import NMREluBenchDataset
import torch
from torch.utils.data import random_split
from omegaconf import DictConfig
import random
from torch.utils.data import Subset

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MyCollator(object):
    def __init__(self, tokenizer, max_length=512, device='cuda'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, examples):
        if self.device:
            device = self.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ret = {}
        smi = []
        nmr1h = []
        nmr13c = []
        for i in examples:
            if "smi" in i.keys():
                smi.append(i["smi"])
            nmr1h.append(i["nmr1h"].cpu().numpy())
            nmr13c.append(i["nmr13c"].cpu().numpy())
        nmr1h = np.ascontiguousarray(np.array(nmr1h))
        nmr13c = np.ascontiguousarray(np.array(nmr13c))
        ret["nmr1h"] = torch.from_numpy(nmr1h).float().to(device)
        ret["nmr13c"] = torch.from_numpy(nmr13c).float().to(device)

        if len(smi) > 0:
            output = self.tokenizer(
                smi,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )  # attention_mask, input_ids, ...
            ret["labels"] = output["input_ids"].contiguous().to(device)
            ret["labels"] = torch.where(ret['labels'] != self.tokenizer.pad_token_id, ret['labels'], -100)
            # the forward function automatically creates the correct decoder_input_ids
            # input["decoder_input_ids"] = output["input_ids"][:, :-1].contiguous().to(device)

        ret['index'] = [i['index'] for i in examples]
        # ret['key'] = [i['key'] for i in examples]
        ret['smi'] = [i['smi'] for i in examples]
        return ret

class NMREluBenchDataModule(pl.LightningDataModule):
    """
    Data module containing a nmr spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    """

    def __init__(
        self,
        split_path: DictConfig,
        datasets: DictConfig,
        use_h: DictConfig,
        use_c: DictConfig,
        use_complete: DictConfig,
        num_points: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__()
        self.split_path = split_path
        self.datasets = datasets
        self.use_h = use_h
        self.use_c = use_c
        self.use_complete = use_complete
        self.num_points = num_points
        self.num_workers = num_workers
        self.batch_size = batch_size
        # self.tokenizer = AutoTokenizer.from_pretrained(f"./cache/{config['model']['smiles_tokenizer']}")
        
    def prepare_data(self):
        """Pre-processing to be executed only on a single main device when using distributed training."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Pre-processing to be executed on every device when using distributed training."""
        self.dataset = hydra.utils.instantiate(self.datasets)
        split = torch.load(self.split_path)
        train_id = split['train_id']
        valid_id = split['valid_id']
        test_id = split['test_id']
        filtered_indices = self.dataset.filtered_indices

        index_map = {idx: i for i, idx in enumerate(filtered_indices)}

        train_id = [index_map[i] for i in train_id if i in index_map]
        valid_id = [index_map[i] for i in valid_id if i in index_map]
        test_id  = [index_map[i] for i in test_id if i in index_map]

        self.train_dataset = Subset(self.dataset, train_id)
        self.val_dataset   = Subset(self.dataset, valid_id)
        self.test_dataset  = Subset(self.dataset, test_id)



        print("dataset", len(self.dataset))
        print("train_dataset", len(self.train_dataset), (self.train_dataset)[0])
        print("val_dataset", len(self.val_dataset), (self.val_dataset)[0])
        print("test_dataset", len(self.test_dataset), (self.test_dataset)[0])
        # self.test_dataset = val_dataset
        # self.val_dataset = hydra.utils.instantiate(self.datasets.test)
        # self.test_dataset = Subset(self.test_dataset, range(min(64, len(self.test_dataset))))
        # self.test_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            shuffle=True,
            num_workers=self.num_workers.train,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            # collate_fn=MyCollator(tokenizer=self.tokenizer),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=self.num_workers.val,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            # collate_fn=MyCollator(tokenizer=self.tokenizer),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size.test,
            shuffle=False,
            num_workers=self.num_workers.test,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            # collate_fn=MyCollator(tokenizer=self.tokenizer),
        )
