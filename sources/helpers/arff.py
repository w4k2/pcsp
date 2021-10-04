import numpy as np
import pandas as pd

from scipy.io import arff


def load_arff_dataset(data_path):
    data_record, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data_record).to_numpy()
    return data[:, :-1], data[:, -1]
