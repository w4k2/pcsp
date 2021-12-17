import numpy as np

from sources.ckm.common import initialize_centers, tolerance
from sklearn.utils import check_random_state
from warnings import warn

def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                warn('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


class PCKMeans:
    def __init__(self, n_clusters=3, max_iter=300, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w

        self.n_iter_ = 0

    def fit(self, X, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize centroids
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)
        cluster_centers = initialize_centers(X, self.n_clusters, 'random', 100)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-4, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        self.n_iter_ = iteration

        return self

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

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
