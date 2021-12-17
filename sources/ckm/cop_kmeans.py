import numpy as np

from sources.ckm.common import initialize_centers, tolerance
from sklearn.utils import check_random_state

from scipy.spatial.distance import cdist as dist

def violates_constraints(i, cluster_index, labels, const_mat):
    for j in np.argwhere(const_mat[i] == 1):
        if 0 < labels[j] != cluster_index:
            return True

    for j in np.argwhere(const_mat[i] == -1):
        if cluster_index == labels[j]:
            return True

    return False


class COPKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, init=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        # Initialization variables
        self.random_state_ = None
        self.cluster_centers_ = None

        # Result variables
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self.random_state_ = None
        self.cluster_centers_ = None
        return self.partial_fit(X, const_mat)

    def partial_fit(self, X, const_mat=None):
        self.labels_ = np.full(X.shape[0], fill_value=-1)
        tol = tolerance(X, self.tol)

        if self.random_state_ is None:
            self.random_state_ = check_random_state(self.random_state)

        # Initialize cluster centers
        if self.cluster_centers_ is None:
            self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, self.random_state)

        # Repeat until convergence or max iters
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self.assign_clusters(X, self.cluster_centers_, const_mat)

            # break if unfeasible assignment
            if -1 in self.labels_:
                break

            prev_cluster_centers = self.cluster_centers_.copy()

            # Estimate means
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0)
                if sum(self.labels_ == i) > 0
                else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - self.cluster_centers_)
            converged = np.allclose(cluster_centers_shift, np.zeros(self.cluster_centers_.shape), atol=tol, rtol=0)

            if converged:
                break

        self.n_iter_ = iteration
        return self

    def assign_clusters(self, X, cluster_centers, const_mat):
        labels = np.full(X.shape[0], fill_value=-1)
        data_indices = np.arange(len(X))
        cdist = dist(X, cluster_centers)

        for i in data_indices:
            for cluster_index in cdist[i].argsort():
                if not violates_constraints(i, cluster_index, labels, const_mat):
                    labels[i] = cluster_index
                    break

        return labels
