import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sources.streams.chunk_generator import ChunkGenerator
from sources.streams.blobs_generator import make_beta_blobs
from sources.helpers.animation import FrameAnimaton
from sources.algorithms.PCSkmeans import PCSKMeans
from sources.algorithms.COPkmeans import COPKMeans
from sources.helpers.constraints import make_constraints
from sources.helpers.arff import load_arff_dataset

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import ecol

from itertools import combinations

from tqdm import tqdm

CHUNK_SIZE = 100
N_CHUNKS = 20
C_RATIO = 0.2

G = 5
W = 2.5

N_CLUSTERS = 24

ESTIMATORS = [
    # KMeans(n_clusters=N_CLUSTERS),
    MiniBatchKMeans(n_clusters=N_CLUSTERS),
    PCSKMeans(n_clusters=N_CLUSTERS),
    COPKMeans(n_clusters=N_CLUSTERS),
]

complexities = []
scores = np.zeros((len(ESTIMATORS), N_CHUNKS))

def main():
    # X, y = make_beta_blobs([[0.5, 0], [-0.5, 0]], n_samples=CHUNK_SIZE * N_CHUNKS, random_state=100)
    # X, y = load_arff_dataset('data/cse/sensor.arff')
    X, y = load_arff_dataset('data/cse/powersupply.arff')

    print(f"n_clusters: {len(np.unique(y))}")

    X = X.astype(float)
    y = LabelEncoder().fit_transform(y)
    # X = PCA(n_components=2).fit_transform(X, y)

    s = ChunkGenerator(X, y, chunk_size=CHUNK_SIZE)

    gif = FrameAnimaton()
    axlim=(np.min(X) - 0.1, np.max(X) + 0.1)
    xlim=(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1)
    ylim=(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1)

    for i, (X_test, y_test) in tqdm(enumerate(s), total=N_CHUNKS):
        if i == N_CHUNKS:
            break

        complexities.append({
            # **ecol.overlapping(X_test, y_test),
            # **ecol.linearity(X_test, y_test),
            # **ecol.neighborhood(X_test, y_test),
            # **ecol.network(X_test, y_test),
            # **ecol.balance(X_test, y_test),
        })
        const = make_constraints(y_test, ratio=C_RATIO, random_state=100)

        fig = plt.figure(figsize=(G * len(ESTIMATORS) + 1, W * G))
        grid = fig.add_gridspec(3, len(ESTIMATORS) + 1)

        # ax = fig.add_subplot(grid[0, 0])
        # ax.set_title("DS")
        # ax.set_ylabel(f"chunk_{i}")
        # ax.scatter(*X_test.T, c=y_test)
        # ax.set_xlim(*axlim)
        # ax.set_ylim(*axlim)

        for j, est in enumerate(ESTIMATORS):
            est_name = type(est).__name__

            if "PCS" in est_name or "COP" in est_name:
                est_ = clone(est)
                est_.fit(X_test, const)
                y_pred = est_.predict(X_test)
            else:
                y_pred = est.fit_predict(X_test, y_test)

            score = adjusted_rand_score(y_test, y_pred)
            scores[j, i] = score

            # ax = fig.add_subplot(grid[0, j + 1])
            # ax.set_title(est_name)
            # ax.scatter(*X_test.T, c=y_pred)
            #
            # if "PCSK" in est_name:
            #     for ci, cj in combinations(range(len(X_test)), 2):
            #         if const[ci, cj] == 1:
            #             ax.plot(X_test[(ci, cj), 0], X_test[(ci, cj), 1], 'g--', alpha=0.3)
            #         elif const[ci, cj] == -1:
            #             ax.plot(X_test[(ci, cj), 0], X_test[(ci, cj), 1], 'r--', alpha=0.1)
            #
            # ax.set_xlim(*axlim)
            # ax.set_ylim(*axlim)
            # plt.setp(ax.get_yticklabels(), visible=False)

        # SCORES
        ax = fig.add_subplot(grid[1, :])
        ax.set_title("Adjusted Rand Score")
        for s in scores:
            ax.plot(np.arange(1, i+2), s[:i+1])

        ax.set_xlim(1, N_CHUNKS)
        ax.set_ylim(-1.1, 1.1)

        ax.grid()
        ax.legend([
            type(est).__name__ for est in ESTIMATORS
        ])

        df = pd.DataFrame(complexities)

        ax = fig.add_subplot(grid[2, :])
        ax.set_title("Difficulties")

        for s in df.to_numpy().T:

            ax.plot(np.arange(1, i+2), s[:i+1])

        ax.set_xlim(1, N_CHUNKS)
        ax.set_ylim(-0.1, 1.1)

        ax.grid()
        ax.legend([ecol.COMPLEXITY_NAMES[n] for n in df.columns])

        plt.tight_layout()
        gif.add_frame()
        plt.savefig("foo.png")
        plt.close()

    gif.export('foo.gif', delay=20)

if __name__ == '__main__':
    main()
