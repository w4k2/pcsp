import numpy as np
from itertools import combinations


# Prepare feasible constrint set
# Max constraints is a number of possible pairwise connections, so $$ \binom{k}{2} = k * (k - 1) / 2 $$.
def make_constraints(y, ratio=1.0, random_state=None, use_matrix=True):
    rng = np.random.default_rng(random_state)
    max_const = (len(y) * (len(y) - 1)) / 2
    n_const = round(max_const * ratio)

    combs = tuple(combinations(range(len(y)), 2))
    indices = sorted(rng.choice(range(len(combs)), n_const, replace=False))
    pairs = [combs[i] for i in indices]

    if use_matrix:
        const_matrix = np.zeros((len(y), len(y)))

        for (i, j) in pairs:
            const_matrix[i, j] = 1 if y[i] == y[j] else -1

        # Consider non-symmetric matrix
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


def const_list_to_cont_mat(ml, cl, n):
    const_matrix = np.zeros((len(y), len(y)))

    for (i, j) in ml:
        const_matrix[i, j] = 1

    for (i, j) in cl:
        if const_matrix[i, j] == 1:
            raise Exception('Inconsistent constraints between %d and %d' % (i, j))

        const_matrix[i, j] = -1

        # Consider non-symmetric matrix
        const_matrix += const_matrix.T

    return const_mat


def const_mat_to_const_list(const_mat):
    ml, cl = [], []

    for i, j in combinations(range(len(const_mat)), 2):
        if const_mat[i, j] == 1:
            ml.append((i, j))
        if const_mat[i, j] == -1:
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
