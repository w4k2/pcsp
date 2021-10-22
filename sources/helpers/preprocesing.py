import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def prepare_X(X):
    if type(X) is pd.DataFrame:
        return X.select_dtypes(include=['float', 'int']).to_numpy()

    return X.astype(np.float)

def prepare_y(y):
    # Simple, easy
    return LabelEncoder().fit_transform(y)
