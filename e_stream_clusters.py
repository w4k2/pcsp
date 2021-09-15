import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_moons
from sklearn.base import clone

from gif import pyplotGIF
from stationary_stream import FoldingStationaryStreamGenerator

from tqdm import tqdm

plt.rcParams.update({'font.size': 7})

N_CHUNKS = 100

ESTIMATORS = [
    Birch(),
    MiniBatchKMeans(),
]

g = 3

scores = np.zeros((N_CHUNKS, len(ESTIMATORS)))

def main():
    gif = pyplotGIF()

    X, y = make_moons(n_samples=10000, random_state=100)
    stream = FoldingStationaryStreamGenerator(X, y, n_chunks=N_CHUNKS)

    n_clusters = len(np.unique(y))
    x_lim = (np.min(stream.X[:, 0]) - 0.1, np.max(stream.X[:, 0]) + 0.1)
    y_lim = (np.min(stream.X[:, 1]) - 0.1, np.max(stream.X[:, 1]) + 0.1)

    for row, (X_test, y_test) in tqdm(enumerate(stream.generate()), total=N_CHUNKS):
        fig = plt.figure(figsize=((1 + len(ESTIMATORS) * g), 2 * g))
        gs = fig.add_gridspec(2, 1 + len(ESTIMATORS))

        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(f"chunk [{row}]")
        ax.scatter(*X_test.T, c=y_test, s=2, alpha=0.7)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        for col, est in enumerate(ESTIMATORS):
            estimator = clone(est)
            try:
                estimator.set_params(n_clusters=n_clusters)
            except Exception:
                pass

            estimator.partial_fit(X_test)

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
        gif.add_frame()
        plt.savefig(f"foo.png")
        plt.close()

    gif.export("animame/e_stream_clusters.gif")


if __name__ == '__main__':
    main()
