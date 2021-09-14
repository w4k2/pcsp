import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import adjusted_rand_score

from sklearn.datasets import make_moons
from stationary_stream import FoldingStationaryStreamGenerator

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

import numpy as np

from tqdm import tqdm

N_CHUNKS = 100

ESTIMATORS = [
    Birch,
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
]

g = 3

scores = np.zeros((N_CHUNKS, len(ESTIMATORS)))

def main():
    X, y = make_moons(n_samples=10000, random_state=100)
    y = LabelEncoder().fit_transform(y)
    n_clusters = len(np.unique(y))

    x_lim = (np.min(X[:, 0] - 0.05), np.max(X[:, 0] + 0.05))
    y_lim = (np.min(X[:, 1] - 0.05), np.max(X[:, 1] + 0.05))

    stream = FoldingStationaryStreamGenerator(X, y, n_chunks=N_CHUNKS)

    for row, (X_test, y_test) in tqdm(enumerate(stream.generate()), total=N_CHUNKS):
        fig, axs = plt.subplots(1, len(ESTIMATORS) + 1, figsize=((1 + len(ESTIMATORS) * g), g))
        axs[0].set_title(f"chunk{row}")
        axs[0].scatter(*X_test.T, c=y_test, alpha=0.5)
        axs[0].set_xlim(*x_lim)
        axs[0].set_ylim(*y_lim)

        for col, est in enumerate(ESTIMATORS):
            estimator = est(n_clusters=n_clusters)
            estimator.fit(X_test)

            if hasattr(estimator, 'labels_'):
                y_pred = estimator.labels_.astype(int)
            else:
                y_pred = estimator.predict(X)

            axs[col+1].set_title(est.__name__)
            axs[col+1].scatter(*X_test.T, c=y_pred)
            axs[col+1].set_xlim(*x_lim)
            axs[col+1].set_ylim(*y_lim)

            scores[row, col] = adjusted_rand_score(y_test, y_pred)

        plt.tight_layout()
        plt.savefig(f"gif/{row}.png")
        plt.close()


    fig, ax = plt.subplots(1, 1)
    for series in scores.T:
        ax.plot(series, alpha=0.7)

    ax.legend([
        est.__name__ for est in ESTIMATORS
    ])

    ax.set_ylim(-1.05, 1.05)
    plt.savefig("foo.png")


if __name__ == '__main__':
    main()
