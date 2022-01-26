import numpy as np

from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import check_random_state

from sources.kmeans import KMeans, PCKMeans, COPKMeans, SCOPKMeans
from sources.kmeans import INIT_KMPP, INIT_RANDOM, INIT_NEIGHBORHOOD
from sources.helpers import make_constraints

import pandas as pd
from pandasgui import show

C_RATIOS = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50
]

INITS = [
    INIT_RANDOM,
    INIT_KMPP,
    INIT_NEIGHBORHOOD
]

ESTIMATORS = [
    KMeans,
    PCKMeans,
    COPKMeans,
    SCOPKMeans,
]

SEED = 1010
N_SAMPLES = 1000
N_CLUSTERS = 2

def main():
    random_state = check_random_state(SEED)
    np_rs = np.random.default_rng(SEED)

    folds = [
        make_moons(n_samples=46, random_state=random_state),
        make_moons(n_samples=46, random_state=random_state),
        make_moons(n_samples=46, random_state=random_state),
        make_moons(n_samples=46, random_state=random_state),
        make_moons(n_samples=46, random_state=random_state)
    ]

    results = []

    for init in INITS:
        for fold_i , (X, y) in enumerate(folds):
            for c_ratio in C_RATIOS:
                const_mat = make_constraints(y, ratio=c_ratio, random_state=np_rs)
                for est in ESTIMATORS:
                    print({"init": init, "c_ratio": c_ratio, "fold": fold_i, "clustering": est.__name__})

                    e = est(n_clusters=N_CLUSTERS, init=init, random_state=SEED)
                    if 'const_mat' in est.fit.__code__.co_varnames:
                        e.fit(X, const_mat=const_mat)
                    else:
                        if init == INIT_NEIGHBORHOOD:
                            continue
                        e.fit(X)

                    score = adjusted_rand_score(y, e.labels_)
                    results.append({"init": init, "c_ratio": c_ratio, "fold": fold_i, "clustering": est.__name__, "n_iters": e.n_iter_, "rand index": score})

    pd.DataFrame.from_dict(results).to_csv("results.csv")

if __name__ == '__main__':
    main()
