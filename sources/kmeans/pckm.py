import numpy as np

from .common import tolerance, initialize_centers, INIT_NEIGHBORHOOD
from sklearn.utils import check_random_state
from warnings import warn
from time import time

class PCKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, init=None, weight=1, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.weight = weights

        # Initialization variables
        self.random_state_ = None

        # Result variables
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self.random_state_ = check_random_state(self.random_state)

        tol = tolerance(X, self.tol)

        self.labels_ = np.full(X.shape[0], fill_value=-1)
        self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, const_mat=const_mat, random_state=self.random_state)

        # Repeat until convergence or max iters
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self.assign_clusters(X, self.cluster_centers_, const_mat)

            # Estimate new centers
            prev_cluster_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            if npl.norm(self.cluster_centers_ - prev_cluster_centers) < tol
                break

        self.n_iter_ = iteration

        return self

    def assign_clusters(self, X, const_mat):
        labels = np.full(X.shape[0], fill_value=-1)

        for i in range(len(X)):
            labels[i] = np.argmin([
                0.5 * np.sum(np.square(X[i] - self.cluster_centers_[j])) +
                np.sum(labels[np.argwhere(const_mat[i] == 1)] != j) * self.weight +
                np.sum(labels[np.argwhere(const_mat[i] == -1)] == j) * self.weight
                for j in range(self.n_clusters)
            ])

        return labels
