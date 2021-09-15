import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.base import clone

from tqdm import tqdm

from constraints import make_constraints
from stationary_stream import FoldingStationaryStreamGenerator
from gif import pyplotGIF
from PCSKMeans import PCSKMeans

plt.rcParams.update({'font.size': 7})

N_CHUNKS = 100

ESTIMATORS = [
    PCSKMeans(),
]

CONSTRAINT_RATIO = 0.1
g = 3

scores = np.zeros((N_CHUNKS, len(ESTIMATORS)))

def main():
    gif = pyplotGIF()

    X, y = make_blobs(n_samples=10000, random_state=100)
    stream = FoldingStationaryStreamGenerator(X, y, n_chunks=N_CHUNKS)

    n_clusters = len(np.unique(y))
    x_lim = (np.min(stream.X[:, 0]) - 0.1, np.max(stream.X[:, 0]) + 0.1)
    y_lim = (np.min(stream.X[:, 1]) - 0.1, np.max(stream.X[:, 1]) + 0.1)

    for row, (X_test, y_test) in tqdm(enumerate(stream.generate()), total=N_CHUNKS):
        const_matrix = make_constraints(y_test, CONSTRAINT_RATIO)

        fig = plt.figure(figsize=((1 + len(ESTIMATORS) * g), 2 * g))
        gs = fig.add_gridspec(2, 1 + len(ESTIMATORS))

        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(f"chunk [{row}]")

        for (i, j) in combinations(range(len(y_test)), 2):
            if const_matrix[i, j] == 1:
                ax.plot(X_test[(i, j), 0], X_test[(i, j), 1], 'g--', alpha=0.2)
            elif const_matrix[i, j] == -1:
                ax.plot(X_test[(i, j), 0], X_test[(i, j), 1], 'r--', alpha=0.2)

        ax.scatter(*X_test.T, c=y_test, s=2, alpha=0.7)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        for col, est in enumerate(ESTIMATORS):
            estimator = clone(est)
            estimator.set_params(n_clusters=n_clusters)
            estimator.fit(X_test, const_matrix)

            if hasattr(estimator, 'labels_'):
                y_pred = estimator.labels_.astype(int)
            else:
                y_pred = estimator.predict(X)

            ax = fig.add_subplot(gs[0, col + 1])

            ax.set_title(type(est).__name__)
            ax.scatter(*X_test.T, c=y_pred, s=2, alpha=0.7)
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            plt.setp(ax.get_yticklabels(), visible=False)

            scores[row, col] = adjusted_rand_score(y_test, y_pred)


        ax = fig.add_subplot(gs[1, :])

        for series in scores.T:
            ax.plot(series[:row], alpha=0.7)

        ax.legend([
            type(est).__name__ for est in ESTIMATORS
        ])

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(-0, N_CHUNKS)
        plt.tight_layout()
        plt.savefig("foo.png")
        gif.add_frame()
        plt.close()

    gif.export("animame/e_constraints.gif")


if __name__ == '__main__':
    main()
