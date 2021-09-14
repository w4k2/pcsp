from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state, _safe_indexing

import numpy as np

class RandomStationaryStreamGenerator:
    def __init__(self, X, y, n_chunks=10, chunk_size=100, random_state=None, replace=False):
        self.X = X
        self.y = y
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = check_random_state(random_state)
        self.replace = replace

        self._clases, self._counts = np.unique(y, return_counts=True)
        self._indices = [
            np.flatnonzero(y == c) for c in self._clases
        ]

    def generate(self):
        ratio = self.chunk_size / len(self.y)
        gen_counts = (self._counts * ratio).astype(int)

        for _ in range(self.n_chunks):
            indices = []

            for c_ind, n_samples in zip(self._indices, gen_counts):
                indices.extend(self.random_state.choice(c_ind, n_samples, replace=self.replace).tolist())

            yield self.X[indices], self.y[indices]



class FoldingStationaryStreamGenerator:
    def __init__(self, X, y, n_chunks=10, random_state=None):
        self.X = X
        self.y = y
        self.n_chunks = n_chunks
        self.random_state = random_state

    def generate(self):
        folding = StratifiedKFold(n_splits=self.n_chunks, random_state=self.random_state)
        for _, ind in folding.split(self.X, self.y):
            yield self.X[ind], self.y[ind]
