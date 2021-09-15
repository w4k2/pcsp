from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from itertools import combinations
from constraints import make_constraints

def main():
    X, y = make_moons(n_samples=20, random_state=100)

    const_matrix = make_constraints(y, 0.2)

    plt.scatter(*X.T, c=y)

    for (i, j) in combinations(range(len(y)), 2):
        if const_matrix[i, j] == 1:
            plt.plot(X[(i, j), 0], X[(i, j), 1], 'g--', alpha=0.5)
        elif const_matrix[i, j] == -1:
            plt.plot(X[(i, j), 0], X[(i, j), 1], 'r--', alpha=0.5)

    plt.savefig("animame/test_constraints.png")


if __name__ == '__main__':
    main()
