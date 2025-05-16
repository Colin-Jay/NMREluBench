from .base_method import BaseMethod
from rdkit import Chem
from tqdm import tqdm
import torch
try:
    from CReSS.infer import ModelInference
except ImportError:
    print("Please install CReSS package first.")

class CReSSMethod(BaseMethod):
    def __init__(self, config_path, pretrain_model_path, device="cpu"):
        super().__init__()
        self.model_inference = ModelInference(
            config_path=config_path,
            pretrain_model_path=pretrain_model_path,
            device=device
        )
    
    def query_transform(self, query_data):
        nmr_types = list(list(query_data.values())[0].keys())
        assert nmr_types == ['C'], f"CReSS only supports C NMR. Got {nmr_types}."
        transformed_data = {}
        for key in tqdm(query_data):
            nmr = query_data[key]['C']
            transformed_data[key] = self.model_inference.nmr_encode(nmr)
        return transformed_data
    
    def library_transform(self, library_data):
        transformed_data = {}
        for key in tqdm(library_data):
            smiles = library_data[key]['smiles']
            rand_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False)
            transformed_data[key] = self.model_inference.smiles_encode(rand_smiles)
        return transformed_data
    
    def score(self, query_item, library_items):
        feature_matrix = torch.cat(library_items, dim=0)
        scores = self.model_inference.get_cos_distance(query_item[0], feature_matrix).tolist()
        return scores
