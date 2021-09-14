from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state, _safe_indexing

import numpy as np

class RandomDynamicStreamGenerator:
    def __init__(self, concepts, drifts, chunk_size=100, n_chunks=100, random_state=None):
        self.concepts = concepts
        self.drifts = np.array(drifts)
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.random_state = check_random_state(random_state)

        self._concepts_meta = []
        for X, y in self.concepts:
            clases, counts = np.unique(y, return_counts=True)
            indices = [
                np.flatnonzero(y == c) for c in clases
            ]

            self._concepts_meta.append((clases, counts, indices))

    def generate(self):
        for proportions, _ in zip(self.drifts.T, range(self.n_chunks)):
            proportions = proportions / np.sum(proportions)
            _X, _y = [], []

            for chunk_ratio, (X, y), (classes, counts, indices) in zip(proportions, self.concepts, self._concepts_meta):
                _indices = []
                n_samples = int(self.chunk_size * chunk_ratio)
                ratio = n_samples / sum(counts)
                gen_counts = (counts * ratio).astype(int)
                for c_ind, n_samples in zip(indices, gen_counts):
                    _indices.extend(self.random_state.choice(c_ind, n_samples, replace=False).tolist())

                _X.append(X[_indices])
                _y.append(y[_indices])

            yield np.concatenate(_X), np.concatenate(_y)
