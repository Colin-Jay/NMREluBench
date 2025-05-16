from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from utils.nmrnet import merge_equi_nmr

class BaseMethod(ABC):
    def __init__(self):
        """
        Initialize
        """
        self.name = self.__class__.__name__

    @abstractmethod
    def query_transform(self, query_data):
        """
        Preprocess query dataset, must be implemented by subclasses.

        :param query_data: Raw query dataset.
        :return: Processed query dataset.
        """
        pass

    @abstractmethod
    def library_transform(self, library_data):
        """
        Preprocess library dataset, must be implemented by subclasses.

        :param library_data: Raw library dataset.
        :return: Processed library dataset.
        """
        pass

    @abstractmethod
    def score(self, query_item, libarary_data):
        """
        Compute scores between query and library, must be implemented by subclasses.

        :param query_item: Query vector.
        :param library_data: Library vector.
        :return: Similarity score.
        """
        pass

    def run(self, dataset, topk):
        """
        Run Querying process.
        
        :param dataset: Dataset instance.
        :param topk: Number of top items to return.
        :return: Dictionary of query results.
        """
        # Preprocess datasets
        print("Preprocessing dataset...")
        query_data = self.query_transform(dataset.query_data)
        library_data = self.library_transform(dataset.library_data)
        library_keys = list(library_data.keys())
        library_items = list(library_data.values())
        # Compute top-k
        print("Querying top-k...")
        query_results = {}
        for query_id, query_item in tqdm(query_data.items()):
            query_results[query_id] = [library_keys[i] for i in self.get_top_k(query_item, library_items, topk)]
        return query_results
        
    def get_top_k(self, query_item, library_items, topk):
        """
        Return top-k most similar library items for a query.

        :param query_item: Query vector.
        :param topk: Number of top items to return.
        :return: List of top-k library item indices.
        """
        scores = self.score(
            query_item,
            library_items,
        )

        # Sort by score in descending order and return top-k indices
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:topk]


class SimulationBaseMethod(BaseMethod):
    def __init__(self, equi_method='topo'):
        """
        Initialize
        """
        super().__init__()
        assert equi_method in ['no', 'topo', 'oracle'], "equi_method must be one of ['no', 'topo', 'oracle']"
        self.equi_method = equi_method
        self.name = f"{self.name}({self.equi_method})"
    
    def library_transform(self, library_data):
        """
        Preprocess library dataset, must be implemented by subclasses.

        :param library_data: Raw library dataset.
        :return: Processed library dataset.
        """
        transformed_data = {}
        for key, value in tqdm(library_data.items()):
            H_shifts = value['atoms_pred'][value['atoms_element'] == 1]
            if self.equi_method == 'no':
                C_shifts = value['atoms_pred'][value['atoms_element'] == 6]
            else:
                C_shifts = np.sort(merge_equi_nmr(value['atoms_pred'], value['atoms_element'], value[f'atoms_equi_{self.equi_method}'], element_id=6))
                
            transformed_data[key] = {'H': np.sort(H_shifts).tolist(),
                                     'C': np.sort(C_shifts).tolist()}
        return self.library_transform2(transformed_data)

    @abstractmethod
    def library_transform2(self, library_data):
        """
        Preprocess library dataset, must be implemented by subclasses.

        :param library_data: Raw library dataset.
        :return: Processed library dataset.
        """
        pass
