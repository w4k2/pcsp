import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster._kmeans import _tolerance

INIT_KMPP = 'k-means++'
INIT_RANDOM = 'random'

# link tolerance from sklearn
tolerance = _tolerance

def initialize_centers(X, n_clusters, init, random_state=None):
    if isinstance(init, np.ndarray) and init.shape == (n_clusters, len(X.T)):
        return np.copy(init)

    if isinstance(init, str) and init == INIT_KMPP:
        return kmeans_plusplus(X, n_clusters, random_state=random_state)[0]

    # else random
    rng = np.random.default_rng(random_state)
    seeds = rng.permutation(len(X))[:n_clusters]
    return np.copy(X[seeds])


def l2_distance(point1, point2):
    return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])
