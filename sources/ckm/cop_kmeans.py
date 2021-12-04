import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
import random
from sklearn.utils import check_random_state
from logging import warn

from .common import initialize_centers, tolerance, l2_distance


def get_const_list(m):
    ml = []
    cl = []

    for i in range(np.shape(m)[0]):
        for j in range(i + 1, np.shape(m)[0]):
            if m[i, j] == 1:
                ml.append((i, j))
            if m[i, j] == -1:
                cl.append((i, j))

    return ml, cl


def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False


def compute_centers(clusters, dataset, k, ml_info):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1

    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i] / float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i])
                              for i in group)
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)),
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)

        for j in range(k - k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers


def get_ml_info(ml, dataset):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]

    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i])
                  for i in groups[j])
              for j in range(len(groups))]

    return groups, scores, centroids


def list_to_graph_TC(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))

    return ml_graph, cl_graph


def list_to_graph(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    for (i, j) in cl:
        add_both(cl_graph, i, j)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))

    return ml_graph, cl_graph


class COPKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, tc=False, init=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.tc = tc
        self.init = init
        self.random_state = random_state

        # Fitting attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.random_state_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self.cluster_centers_ = None
        self.labels_ = None
        return self.partial_fit(X, const_mat)

    def partial_fit(self, X, const_mat=None):
        self.random_state_ = check_random_state(self.random_state)

        if self.cluster_centers_ is None:
            self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, self.random_state)

        ml, cl = get_const_list(const_mat)

        if self.tc:
            ml, cl = list_to_graph_TC(ml, cl, len(X))
        else:
            ml, cl = list_to_graph(ml, cl, len(X))

        ml_info = get_ml_info(ml, X)
        tol = tolerance(X, self.tol)

        centers = self.cluster_centers_
        clusters = [-1] * len(X)

        for it in range(self.max_iter):
            self.labels_ = [-1] * len(X)
            for i, d in enumerate(X):
                indices, _ = closest_clusters(centers, d)
                counter = 0
                if self.labels_[i] == -1:
                    found_cluster = False
                    while (not found_cluster) and counter < len(indices):
                        index = indices[counter]
                        if not violate_constraints(i, index, self.labels_, ml, cl):
                            found_cluster = True
                            self.labels_[i] = index
                            for j in ml[i]:
                                self.labels_[j] = index
                        counter += 1

                    if not found_cluster:
                        self.n_iter_ = it
                        # warn(f"No feasible assignation. Iteration: {i}")
                        return self

            self.labels_, self.cluster_centers_ = compute_centers(self.labels_, X, self.n_clusters, ml_info)
            shift = sum(l2_distance(centers[i], self.cluster_centers_[i]) for i in range(self.n_clusters))
            if shift <= tol:
                break

            clusters = self.labels_
            centers = self.cluster_centers_

        self.n_iter_ = it
        return self

    def predict(self, X):
        return self.labels_
