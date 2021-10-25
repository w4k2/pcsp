import numpy as np

from sources.ckm.common import initialize_centers, tolerance, l2_distance
from sklearn.utils import check_random_state


def violates_constraints(i, cluster_index, labels, const_mat):
    for j in np.argwhere(const_mat[i] == 1):
        if 0 < labels[j] != cluster_index:
            return True

    for j in np.argwhere(const_mat[i] == -1):
        if cluster_index == labels[j]:
            return True

    return False


# MiniBatch approach
class MiniBatchCOPKmeans:
    def __init__(self, n_clusters=2, max_iter=100, batch_size=100, tol=1e-3, init='random', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
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
        self.random_state_ = check_random_state(self.random_state)
        self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, self.random_state)
        return self.partial_fit(X, const_mat)

    def partial_fit(self, X, const_mat=None):
        self.labels_ = np.full(X.shape[0], fill_value=-1)
        tol = tolerance(X, self.tol)

        if self.random_state_ is None:
            self.random_state_ = check_random_state(self.random_state)

        # Initialize cluster centers
        if self.cluster_centers_ is None:
            self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, self.random_state)

        cluster_centers = self.cluster_centers_

        # Repeat until convergence or max iters
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self.assign_clusters(X, cluster_centers, l2_distance, const_mat)

            # Estimate means
            cluster_centers = np.array([
                X[labels == i].mean(axis=0)
                if sum(labels == i) > 0
                else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=tol, rtol=0)

            if converged:
                break

        self.n_iter_ = iteration
        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def assign_clusters(self, X, cluster_centers, dist, const_mat):
        labels = np.full(X.shape[0], fill_value=-1)

        data_indices = np.arange(len(X))
        self.random_state_.shuffle(data_indices)

        for i in data_indices:
            distances = np.array([
                dist(X[i], c) for c in cluster_centers
            ])

            for cluster_index in distances.argsort():
                if not violates_constraints(i, cluster_index, labels, const_mat):
                    labels[i] = cluster_index
                    break

        return labels
