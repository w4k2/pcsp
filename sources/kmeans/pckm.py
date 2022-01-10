import numpy as np

from .common import tolerance, initialize_centers, INIT_NEIGHBORHOOD
from sklearn.utils import check_random_state
from warnings import warn
from time import time

class PCKMeans:
    def __init__(self, n_clusters=3, init=INIT_NEIGHBORHOOD, max_iter=300, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w

        # Initialization variables
        self.random_state_ = None
        self.cluster_centers_ = None

        # Result variables
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self.labels_ = np.full(X.shape[0], fill_value=-1)
        tol = tolerance(X, self.tol)

        if self.random_state_ is None:
            self.random_state_ = check_random_state(self.random_state)

        # Initialize cluster centers
        if self.cluster_centers_ is None:
            self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, const_mat=const_mat, random_state=self.random_state)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self._assign_clusters(X, self.cluster_centers_, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = self._get_cluster_centers(X, self.labels_)

            # Check for convergence
            difference = (prev_cluster_centers - self.cluster_centers_)
            converged = np.allclose(difference, np.zeros(self.cluster_centers_.shape), atol=1e-4, rtol=0)

            if converged: break

        self.n_iter_ = iteration

        return self

    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            labels[x_i] = np.argmin([self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            print("Empty clusters")

        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
