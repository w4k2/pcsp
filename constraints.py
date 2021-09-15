from sklearn.utils import check_random_state
import numpy as np
from itertools import combinations

def make_constraints(y, ratio=1.0, random_state=None):
    rs = check_random_state(random_state)
    # Max constraints is a number of possible pairwise connections, so $$ \binom{k}{2} = k * k - 1 / 2 $$.
    max_const = (len(y) * (len(y) - 1)) / 2
    n_const = round(max_const * ratio)

    # isn't that pythonic
    # source: https://docs.python.org/3/library/itertools.html
    combs = tuple(combinations(range(len(y)), 2))
    indices = sorted(rs.choice(range(len(combs)), n_const, replace=False))
    pairs = [combs[i] for i in indices]

    const_matrix = np.zeros((len(y), len(y)))

    for (i, j) in pairs:
        const_matrix[i, j] = 1 if y[i] == y[j] else -1

    return const_matrix
