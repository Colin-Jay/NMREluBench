from .base_method import SimulationBaseMethod
import numpy as np
from tqdm import tqdm

class VectorMethod(SimulationBaseMethod):
    def __init__(self, dim=128, sigma_H=0.5, sigma_C=5, range_H=(-1,15), range_C=(-10,230), **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.sigma = {'H': sigma_H, 'C': sigma_C}
        self.range = {'H': range_H, 'C': range_C}
        self.ni = {'H': np.linspace(range_H[0], range_H[1], dim), 'C': np.linspace(range_C[0], range_C[1], dim)}
        interval = {'H': self.ni['H'][1] - self.ni['H'][0], 'C': self.ni['C'][1] - self.ni['C'][0]}
        self.coef = {'H': interval['H'] / (np.sqrt(2*np.pi) * sigma_H), 'C': interval['C'] / (np.sqrt(2*np.pi) * sigma_C)}
        
    def query_transform(self, query_data):
        transformed_data = {}
        for key, sub_dict in tqdm(query_data.items()):
            transformed_data[key] = {}
            for nmr_type, value_list in sub_dict.items():
                if nmr_type in ['H', 'C']:
                    transformed_data[key][nmr_type] = set2vec(np.array(value_list), self.dim, 
                                                              self.sigma[nmr_type], self.ni[nmr_type], self.coef[nmr_type])
        return transformed_data
    
    def library_transform2(self, library_data):
        return self.query_transform(library_data)
    
    def score(self, query_item, library_items):
        query_vectors_H = np.array([query_item["H"]]) if "H" in query_item else None
        query_vectors_C = np.array([query_item["C"]]) if "C" in query_item else None

        library_vectors_H = np.array([item["H"] for item in library_items if "H" in item]) if "H" in query_item else None
        library_vectors_C = np.array([item["C"] for item in library_items if "C" in item]) if "C" in query_item else None

        scores = np.zeros(len(library_items))

        if query_vectors_H is not None:
            scores += np.dot(query_vectors_H, library_vectors_H.T).flatten()

        if query_vectors_C is not None:
            scores += np.dot(query_vectors_C, library_vectors_C.T).flatten()

        return scores.tolist()


def set2vec(set_list, dim, sigma, ni, coef, normalize=True):
    """
    Convert a set of values into a vector representation using Gaussian kernels.

    :param set_list: List of numpy arrays.
    :param dim: Dimensionality of the vector.
    :param sigma: Standard deviation for the Gaussian kernel.
    :param ni: Grid points for the vectorization.
    :param coef: Coefficient for normalization.
    :param normalize: Whether to normalize the resulting vectors.
    :return: vectord representation of the input set.
    """
    if isinstance(set_list, np.ndarray):
        return set2vec([set_list], dim, sigma, ni, coef, normalize)[0]
    # set_list: list of np.array
    assert isinstance(set_list, list)
    # Gaussian kernel
    result = [coef * np.exp(-(np.abs(item[:, np.newaxis] - ni) / sigma) ** 2 / 2).sum(axis=0) for item in set_list]
    result = np.array(result)
    if normalize:
        return normalize_vectors(result).astype(np.float32)
    else:
        return result.astype(np.float32)

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length.

    :param vectors: Input vectors.
    :return: Normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms