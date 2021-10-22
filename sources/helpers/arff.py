import numpy as np
import pandas as pd
import os

from scipy.io import arff

DATASET_CACHE = "__dscache"
if not os.path.isdir:
    os.mkdir(DATASET_CACHE)


def load_arff_dataset(data_path, cache=DATASET_CACHE):
    data_record, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data_record)
    return data.iloc[:, :-1], data.iloc[:, -1]
