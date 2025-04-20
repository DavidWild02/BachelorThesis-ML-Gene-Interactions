import scipy
from typing import Iterator, Tuple
from torch.utils.data import Sampler
import numpy as np
import pandas as pd

class RandomMappingSampler(Sampler):
    def __init__(self, data_a: np.ndarray | pd.DataFrame, data_b: np.ndarray | pd.DataFrame):
        self._data_A = data_a
        self._data_B = data_b
        super().__init__()

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        while True:
            index_A = np.random.randint(0, self._data_A.shape[0])
            index_B = np.random.randint(0, self._data_B.shape[0])
            # a sampler should just return the indices and not the samples
            yield index_A, index_B


class WeightedDistanceMappingSampler(Sampler):
    def __init__(self, data_a, data_b, distance, variance, delta):
        self._data_a = data_a
        self._data_b = data_b
        self._delta = delta
        self._kernel = scipy.stats.norm(distance, np.sqrt(variance))
        self._max_likelihood = self._kernel.pdf(distance) 
        self._random_sampler = iter(RandomMappingSampler(self._data_a, self._data_b))
        super().__init__()

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        while True:
            index_a, index_b = next(self._random_sampler)

            distance = np.abs(self._data_a[index_a] - self._data_b[index_b]) # l1-norm
            probability = self._kernel.cdf(distance + self._delta) - self._kernel.cdf(distance - self._delta)
            if np.random.uniform(0., 1.) < probability:
                continue

            yield index_a, index_b