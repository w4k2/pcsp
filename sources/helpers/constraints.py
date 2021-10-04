import numpy as np
from itertools import combinations

def make_constraints(y, ratio=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    # Max constraints is a number of possible pairwise connections, so $$ \binom{k}{2} = k * k - 1 / 2 $$.
    max_const = (len(y) * (len(y) - 1)) / 2
    n_const = round(max_const * ratio)

    # isn't that pythonic
    # source: https://docs.python.org/3/library/itertools.html
    combs = tuple(combinations(range(len(y)), 2))
    indices = sorted(rng.choice(range(len(combs)), n_const, replace=False))
    pairs = [combs[i] for i in indices]

    const_matrix = np.zeros((len(y), len(y)))

    for (i, j) in pairs:
        const_matrix[i, j] = 1 if y[i] == y[j] else -1

    # Consider not a symmetric matrix
    const_matrix += const_matrix.T

    return const_matrix

def main():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=100)
    const = make_constraints(y, ratio=0.01)

    plt.figure(figsize=(4, 4))
    ax = plt.axes()

    ax.scatter(*X.T, c=y)

    for i, j in combinations(range(len(X)), 2):
        if const[i, j] == 1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'g--', alpha=0.3)
        elif const[i, j] == -1:
            ax.plot(X[(i, j), 0], X[(i, j), 1], 'r--', alpha=0.1)

    plt.tight_layout()
    plt.savefig("foo.png")
    plt.close()

if __name__ == '__main__':
    main()
