from .common import initialize_centers, tolerance
from numpy import linalg as npl
import numpy as np
from scipy.spatial.distance import cdist as dist


class KMeans:
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

    def fit(self, X):
        tol = tolerance(X, self.tol)

        self.labels_ = np.full(X.shape[0], fill_value=-1)
        self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, random_state=self.random_state)

        # Repeat until convergence or max iters
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self.assign_clusters(X)

            # Estimate new centers
            prev_cluster_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            if npl.norm(self.cluster_centers_ - prev_cluster_centers) < tol:
                break

        self.n_iter_ = iteration

        return self

    def assign_clusters(self, X):
        cdist = dist(X, self.cluster_centers_)
        labels = np.argmin(cdist, axis=1)
        return labels
