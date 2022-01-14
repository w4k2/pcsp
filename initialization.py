import numpy as np

from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from sources.kmeans import KMeans, PCKMeans, COPKMeans, SCOPKMeans
from sources.kmeans import INIT_KMPP, INIT_RANDOM, INIT_NEIGHBORHOOD
from sources.helpers import make_constraints

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
    # KMeans,
    # PCKMeans,
    # COPKMeans,
    SCOPKMeans,
]

SEED = 1010

N_CLUSTERS = 2

def main():
    X, y = make_moons(n_samples=200, random_state=100)
    for c_ratio in C_RATIOS:
        print('ratio', c_ratio)
        const_mat = make_constraints(y, ratio=c_ratio, random_state=100)

        for init in INITS:
            print('init', init)
            for est in ESTIMATORS:
                e = est(n_clusters=N_CLUSTERS, init=init, random_state=SEED)
                if 'const_mat' in est.fit.__code__.co_varnames:
                    e.fit(X, const_mat=const_mat)
                else:
                    e.fit(X)

                print(est.__name__)
                score = adjusted_rand_score(y, e.labels_)
                print(e.n_iter_, '|', score)

if __name__ == '__main__':
    main()
