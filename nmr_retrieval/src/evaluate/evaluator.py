import sys
import os
from src.data import RetrievalLMDBDataset
from src.method import BaseMethod

from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool
from functools import partial
from myopic_mces import MCES

class NMRRetrievalEvaluator:
    def __init__(self, k_list, metric_list):
        """
        Initialize the evaluator.

        :param k_list: A list containing multiple k values.
        :param metric_list: A list of metrics to evaluate.
        """
        self.k_list = k_list
        self.metric_list = metric_list

    def evaluate(self, dataset: RetrievalLMDBDataset, method: BaseMethod):
        """
        Evaluate the performance of the retrieval method.

        :param method: An instance of a retrieval method implementing BaseMethod.
        :param dataset: The dataset to evaluate on.
        :return: Evaluation results (e.g., top-k recall).
        """
        print('-' * 20)
        print("Running method:", method.name)
        query_results = method.run(dataset, topk=max(self.k_list))

        # Calculate metrics
        print("Calculating metrics...")
        metric_results = {}
        for metric in self.metric_list:
            metric_results[metric] = self.evaluate_metric(metric, dataset, query_results)
            
        print("Results:", metric_results)

        return metric_results

    def evaluate_metric(self, metric, dataset, query_results):
        """
        Calculate the specified evaluation metric.

        :param metric: The name of the metric.
        :param dataset: The dataset instance.
        :param query_results: The query results.
        :return: The value of the specified metric.
        """
        func_map = {
            "recall": match,
            "MCES": MCES_distance,
        }
        pooling_map = {
            "recall": lambda x: max(x),
            "MCES": lambda x: min(x),
        }
        
        # Get SMILES from query_results
        func = func_map[metric]
        pooling = pooling_map[metric]
        
        score_list = [0] * len(self.k_list)
        for query_id, query_result in tqdm(query_results.items()):
            # Get top k SMILES
            query_smiles = dataset.data[query_id]["smiles"]
            top_k_smiles = [dataset.data[res_id]["smiles"] for res_id in query_result[:max(self.k_list)]]
            # Calculate score
            scores = [func(query_smiles, smi) for smi in top_k_smiles]
            # with Pool() as pool:
            #     scores = pool.map(partial(func, smi2=query_smiles), top_k_smiles)
            for i, k in enumerate(self.k_list):
                score_list[i] += pooling(scores[:k])
                
        # Calculate average
        score_list = [score / len(query_results) for score in score_list]
        
        return score_list


def match(smi1, smi2):
    cononical_smi1 = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi1)), canonical=True)
    cononical_smi2 = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi2)), canonical=True)
    return 1 if cononical_smi1 == cononical_smi2 else 0

def MCES_distance(smi1, smi2):
    n_mol1 = Chem.RemoveHs(Chem.MolFromSmiles(smi1)).GetNumAtoms()
    n_mol2 = Chem.RemoveHs(Chem.MolFromSmiles(smi2)).GetNumAtoms()
    assert n_mol1 > 0 and n_mol2 > 0, f"Invalid SMILES: {smi1}, {smi2}"
    if smi1 == smi2:
        return 0
    if n_mol1 == 1 and n_mol2 == 1:
        return 2
    # Suppress the output of the MCES function
    with open(os.devnull, 'w') as devnull:
        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            mces = MCES(smi1, smi2, threshold=15)
        finally:
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
    return mces[1]