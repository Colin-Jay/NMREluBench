from .base_method import BaseMethod
from numpy import random

class RandomMethod(BaseMethod):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
    def query_transform(self, query_data):
        return query_data
    
    def library_transform(self, library_data):
        return library_data
    
    def score(self, query_item, library_items):
        scores = random.rand(len(library_items))
        return scores
