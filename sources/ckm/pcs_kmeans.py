import numpy as np
from scipy import spatial as sdist
import copy

from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from .common import initialize_centers, tolerance, l2_distance


def binary_search_delta(obj, s, max_iter=200):
    # Norma de Frobenius por defecto
    frob_norm = np.linalg.norm(obj)

    if frob_norm == 0 or np.sum(np.abs(obj/frob_norm)) <= s:
        return 0
    else:
        lam1 = 0
        lam2 = np.max(np.abs(obj)) - 1e-5
        iters = 0

        while iters < max_iter and (lam2 - lam1) > 1e-4:
            su = np.sign(obj) * (np.abs(obj)-((lam1+lam2)/2)).clip(min = 0)

            if np.sum(abs(su/np.linalg.norm(su,2))) < s:
                lam2 = (lam1+lam2) / 2
            else:
                lam1 = (lam1+lam2) / 2

        iters += 1

        return (lam1+lam2)/2


class PCSKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, sparsity=1.1, tol=1e-4, max_iter=20000, init=None, random_state=None):
        # Parameters
        self.n_clusters = n_clusters
        self.sparsity = sparsity
        self.tol = tol
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state

        # Fitting attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.weights_ = None
        self.random_state_ = None

    def fit(self, X, const_mat=None):
        self.cluster_centers_ = None
        self.labels_ = None
        self.weights_ = None
        return self.partial_fit(X, const_mat)


    def partial_fit(self, X, const_mat=None):
        self.random_state_ = check_random_state(self.random_state)

        if self.cluster_centers_ is None:
            self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, self.random_state)

        # n = number of instances / d = number of features
        n, d = X.shape
        tol = tolerance(X, self.tol)

        max_distance = np.power(np.max(X, axis=0) - np.min(X, axis=0), 2)
        self.weights_ = np.ones(d, dtype=np.float) * np.sqrt(d)

        # Compute global centroid
        global_centroid = np.mean(X, axis=0)
        distance_to_global_centroid = np.sum(np.power(X - global_centroid, 2), axis = 0)

        # Initialize partition
        self.labels_ = np.ones(n) * -1

        iters = 0
        shift = -1

        while iters < self.max_iter:
            iters += 1
            # Assign each instance to its closest cluster centroid
            for i in range(n):
                instance = X[i,:]
                # Compute weigthed squared euclidean distances
                weighted_squared_diffs = np.sum(np.power(self.cluster_centers_ - instance, 2) * self.weights_, axis=1)

                # Compute penalties
                penalties = np.zeros(self.n_clusters)

                for l in range(self.n_clusters):
                    for j in range(n):
                        # if the second instance has a label
                        if self.labels_[j] != -1:
                            # if ML(i,j) and instance i is going to be assigned to a label other than the label of instance j
                            if const_mat[i,j] == 1 and l != self.labels_[j]:
                                penalties[l] += np.sum(np.power(instance - X[j,:], 2) * self.weights_)
                            # if ML(i,j) and instance i is going to be assigned to a label equal to the label of instance j
                            if const_mat[i,j] == -1 and l == self.labels_[j]:
                                penalties[l] += np.sum((max_distance - np.power(instance - X[j,:], 2)) * self.weights_)

                self.labels_[i] = np.argmin(weighted_squared_diffs + penalties)

            # Recompute centroids
            old_centroids = copy.deepcopy(self.cluster_centers_)
            for i in range(self.n_clusters):
                cluster = X[np.where(self.labels_ == i)[0], :]
                if cluster.shape[0] > 0:
                    self.cluster_centers_[i,:] = np.mean(cluster, axis=0)

            # Update weights
            within_cluster_distances = np.zeros(d)
            gammas = np.zeros(d)

            for i in range(d):
                for l in range(self.n_clusters):
                    cluster_indices = np.where(self.labels_ == l)[0]
                    cluster = X[cluster_indices,:]
                    within_cluster_distance = np.power(cluster[:,i] - self.cluster_centers_[l,i], 2)

                    # Compute penalties
                    penalties = np.zeros(len(cluster_indices))
                    for j in range(len(cluster_indices)):
                        instance_index = cluster_indices[j]
                        for m in range(n):
                            if const_mat[instance_index,m] == 1 and self.labels_[instance_index] != self.labels_[m]:
                                penalties[j] += ((X[instance_index,i] - X[m,i])**2)

                            if const_mat[instance_index,m] == -1 and self.labels_[instance_index] == self.labels_[m]:
                                penalties[j] += (max_distance[i] - (X[instance_index,i] - X[m,i])**2)

                within_cluster_distances[i] = np.sum(within_cluster_distance)
                gammas[i] = distance_to_global_centroid[i] - np.sum(within_cluster_distance) + np.sum(penalties)

            delta = binary_search_delta(distance_to_global_centroid - within_cluster_distances, self.sparsity)
            self.weights_ = (np.sign(gammas) * (np.abs(gammas) - delta).clip(min = 0)) / np.linalg.norm(np.sign(gammas) * (np.abs(gammas) - delta))

            # Compute centroid shift for stopping criteria
            shift = sum(l2_distance(self.cluster_centers_[i], self.cluster_centers_[i]) for i in range(self.n_clusters))

            if shift <= tol:
                break

        return self

    def predict(self, X):
        # How about checking for same X?
        return self.labels_
