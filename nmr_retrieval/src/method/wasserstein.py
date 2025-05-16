from .base_method import SimulationBaseMethod

from scipy.stats import wasserstein_distance
from multiprocessing import Pool

class WassersteinMethod(SimulationBaseMethod):
    def __init__(self, weight_H=1, weight_C=0.2, centering=False, **kwargs):
        """
        Initialize the Wasserstein method class.
        """
        super().__init__(**kwargs)
        self.weight = {'H': weight_H, 'C': weight_C}
        self.centering = centering
        
    def query_transform(self, query_data):
        return query_data
    
    def library_transform2(self, library_data):
        return library_data
    
    def score_single(self, query_item, library_item):
        score = 0
        for nmr_type in query_item:
            score += wasserstein_score(query_item[nmr_type], library_item[nmr_type], self.centering) * self.weight[nmr_type]
        return score
        
    def score(self, query_item, library_items):
        """
        Compute similarity scores between a query item and multiple library items.
        """
        with Pool() as pool:
            scores = pool.starmap(self.score_single, [(query_item, lib_item) for lib_item in library_items])
        return scores


def wasserstein_score(list1, list2, centering):
    """
    Compute the Wasserstein distance.

    :param centering: Whether to center the data.
    :param list1: First list of values.
    :param list2: Second list of values.
    :return: Negative Wasserstein distance.
    """
    if not list1 or not list2:  # Check if either list is empty
        return - float('inf')
    if centering:
        list1 = [x - sum(list1) / len(list1) + sum(list2) / len(list2) for x in list1]
    return - wasserstein_distance(list1, list2)
