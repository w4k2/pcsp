from scipy.io import arff
import numpy as np
import pandas as pd


def parse_arff_dataset(data_path):
    data_record, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data_record).to_numpy()
    return data[:, :-1], data[:, -1]


class RealStreamGenerator:
    def __init__(self, data_path, chunk_size=100):
        self.data_path = data_path
        self.chunk_size = chunk_size

        self.X, self.y = parse_arff_dataset(self.data_path)
        self._n_chunks = len(self.X) // self.chunk_size

    def generate(self):
        last = 0
        new = self.chunk_size

        for _ in range(self._n_chunks):
            yield self.X[last:new], self.y[last:new]
            last = new
            new += self.chunk_size
