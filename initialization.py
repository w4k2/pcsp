import numpy as np
from collections import OrderedDict
from sources.kmeans import COPKMeans, PCKMeans
from sklearn.datasets import make_moons
from sources.helpers.constraints import make_constraints, const_mat_to_const_list

C_RATIO = 0.2

def main():
    X, y = make_moons(n_samples=200, random_state=100)
    n_clusters = len(np.unique(y))
    const_mat = make_constraints(y, ratio=C_RATIO, random_state=100)

    e = COPKMeans(n_clusters=n_clusters, init="neighborhood")
    e.fit(X, const_mat)

    ml, cl = const_mat_to_const_list(const_mat)
    e = PCKMeans(n_clusters=n_clusters)
    e.fit(X, ml, cl)

if __name__ == '__main__':
    main()
