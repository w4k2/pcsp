import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from scipy.io import arff

DEFAULT_DATASET_STORAGE = "_datasets_npy"
if not os.path.isdir(DEFAULT_DATASET_STORAGE):
    os.mkdir(DEFAULT_DATASET_STORAGE)


def load_arff_dataset(data_path):
    data_record, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data_record)
    return data.iloc[:, :-1], data.iloc[:, -1]


def prepare_X(X):
    if type(X) is pd.DataFrame:
        return X.select_dtypes(include=['float', 'int']).to_numpy()

    return X.astype(np.float)


def prepare_y(y):
    # Simple, easy
    return LabelEncoder().fit_transform(y)


def save_npy(X, y, dataset_name, datasets_path=DEFAULT_DATASET_STORAGE):
    np.save(os.path.join(datasets_path, f"{dataset_name}_X.npy"), X)
    np.save(os.path.join(datasets_path, f"{dataset_name}_y.npy"), y)


def load_npy(dataset_name, datasets_path=DEFAULT_DATASET_STORAGE):
    X = np.load(os.path.join(datasets_path, f"{dataset_name}_X.npy"))
    y = np.load(os.path.join(datasets_path, f"{dataset_name}_y.npy"))
    return X, y
