import numpy as np
from itertools import combinations

# Prepare feasible constrint set
def make_constraints(y, ratio=1.0, random_state=None, use_matrix=False):
    rng = np.random.default_rng(random_state)
    # Max constraints is a number of possible pairwise connections, so $$ \binom{k}{2} = k * (k - 1) / 2 $$.
    max_const = (len(y) * (len(y) - 1)) / 2
    n_const = round(max_const * ratio)

    combs = tuple(combinations(range(len(y)), 2))
    indices = sorted(rng.choice(range(len(combs)), n_const, replace=False))
    pairs = [combs[i] for i in indices]

    if use_matrix:
        const_matrix = np.zeros((len(y), len(y)))

        for (i, j) in pairs:
            const_matrix[i, j] = 1 if y[i] == y[j] else -1

        # Consider not a symmetric matrix
        const_matrix += const_matrix.T

        return const_matrix
    else:
        ml, cl = [], []

        for (i, j) in pairs:
            if y[i] == y[j]:
                ml.append((i, j))
            else:
                cl.append((i, j))

        return ml, cl


def check_unsatisfied_constraints(const_mat, y_pred):
    chk_mat = np.zeros_like(const_mat)

    for i, j in combinations(range(len(y_pred)), 2):
        if const_mat[i, j] == 1:
            if y_pred[i] != y_pred[j]:
                chk_mat[i, j] = 1
        elif const_mat[i, j] == -1:
            if y_pred[i] == y_pred[j]:
                chk_mat[i, j] = 1

    return chk_mat


def const_list_to_graph(ml, cl, n):
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
                raise Exception('Inconsistent constraints between %d and %d' % (i, j))

    return ml_graph, cl_graph


def const_mat_to_const_list(const_mat):
    ml, cl = [], []

    for i, j in combinations(range(len(const_mat)), 2):
        if const_mat[i, j] == 1:
            ml.append((i, j))
        if const_mat[i, j] == -1:
            cl.append((i, j))

    return ml, cl


def main():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=20, random_state=100)
    const_mat = make_constraints(y, ratio=0.05, use_matrix=True)

    y_pred = KMeans(n_clusters=len(np.unique(y))).fit_predict(X)
    chk_mat = check_unsatisfied_constraints(const_mat, y_pred)
    C = np.count_nonzero(const_mat) // 2
    U = chk_mat.sum()

    print(U)
    print(C)
    print(U / C)

    fig = plt.figure(figsize=(6, 5))
    spec = fig.add_gridspec(ncols=1, nrows=1)

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(*X.T, c=y, s=40)

    for i, j in combinations(range(len(X)), 2):
        if const_mat[i, j] == 1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'g--', alpha=0.5)
        elif const_mat[i, j] == -1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'r--', alpha=0.3)

    # ax = fig.add_subplot(spec[0, 1])
    # ax.scatter(*X.T, c=y_pred)

    plt.tight_layout()
    plt.savefig("foo.png")
    plt.close()

if __name__ == '__main__':
    main()
