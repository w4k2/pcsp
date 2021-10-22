import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sources.streams.chunk_generator import ChunkGenerator
from sources.helpers.constraints import make_constraints
from sources.helpers.arff import load_arff_dataset
from sources.streams.blobs_generator import make_beta_blobs
from sources.helpers.preprocesing import prepare_X, prepare_y
from sources.helpers.animation import FrameAnimaton

from sources.ckm.cop_kmeans import COPKMeans
from sources.ckm.pcs_kmeans import PCSKMeans
from sources.ckm.mini_batch.cop_kmeans import MiniBatchCOPKmeans

from sklearn.metrics import adjusted_rand_score

from tqdm import tqdm

CHUNK_SIZE = 200
R_N_CHUNKS = 50
C_RATIO = 0.05
MAX_CHUNKS = 30

G = 5
W = 1.5


def make_stream_gif(X, y, chunk_size, out_file="stream.gif"):
    print(len(X))
    print(chunk_size * MAX_CHUNKS)

    n_samples = min(len(X), chunk_size * MAX_CHUNKS)
    X = X[:n_samples]
    y = y[:n_samples]

    s = ChunkGenerator(X, y, chunk_size=chunk_size)
    animation = FrameAnimaton()

    # ax_lims = X.min() - 0.1, X.max() + 0.1

    for i, (X_test, y_test) in enumerate(s):
        if i > MAX_CHUNKS:
            break

        fig = plt.figure()
        plt.title(f"{i}")

        if X_test.shape[1] > 2:
            print("Applying reduction")
            from sklearn.manifold import Isomap as tranformation
            X_test = tranformation(n_components=2).fit_transform(X_test, y)

        plt.scatter(*X_test.T, c=y_test, s=2)
        # plt.xlim(*ax_lims)
        # plt.ylim(*ax_lims)
        plt.tight_layout()
        animation.add_frame(fig)
        plt.close(fig)

    animation.export(out_file)


def main():
    X, y = load_arff_dataset('data/cse/kddcup99.arff')
    # X, y = load_arff_dataset('data/cse/sensor.arff')
    # X, y = load_arff_dataset('data/cse/powersupply.arff')
    # X, y = make_beta_blobs([[3.1, 0], [2.1, 0], [1.1, 0], [1.1, 1]], radius=[0.5, 0.5, 0.5, 0.5],
    #                        n_samples=CHUNK_SIZE * R_N_CHUNKS, random_state=100)

    X = prepare_X(X)
    y = prepare_y(y)

    print("Preparing animation")
    make_stream_gif(X, y, CHUNK_SIZE, "animation.gif")
    exit()

    stream = ChunkGenerator(X, y, chunk_size=CHUNK_SIZE)
    N_CHUNKS = min(len(X) // stream.chunk_size, MAX_CHUNKS)

    N_CLUSTERS = len(np.unique(y))

    ESTIMATORS = [
        PCSKMeans(n_clusters=N_CLUSTERS, random_state=100),
        COPKMeans(n_clusters=N_CLUSTERS, random_state=100),
        MiniBatchCOPKmeans(n_clusters=N_CLUSTERS, random_state=100),
    ]

    ESTIMATORS_N = [
        "PCSKMeans",
        "COPKMeans",
        "ShuffledCOPKmeans",
    ]

    scores = np.zeros((len(ESTIMATORS), N_CHUNKS))
    etimes = np.zeros((len(ESTIMATORS), N_CHUNKS))

    for i, (X_test, y_test) in tqdm(enumerate(stream), total=N_CHUNKS):
        if i == N_CHUNKS:
            break

        const_mat = make_constraints(y_test, ratio=C_RATIO, random_state=100)

        for j, est in enumerate(ESTIMATORS):
            s_time = time.time()
            est.partial_fit(X_test, const_mat)
            y_pred = est.labels_
            e_time = time.time()
            scores[j, i] = adjusted_rand_score(y_test, y_pred)
            etimes[j, i] = e_time - s_time

        fig = plt.figure(figsize=(G * 3 + 1, W * G))
        grid = fig.add_gridspec(2, 3 + 1)

        # SCORES
        ax = fig.add_subplot(grid[0, :], figsize=(2, 2))
        ax.set_title("Performance")
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_xlabel("Chunks")

        for s in scores:
            ax.plot(np.arange(1, i + 2), s[:i + 1], linewidth=1.2, alpha=0.3)

        for s in scores:
            rolling_mean = pd.Series(s[:i + 1]).rolling(window=5).mean()
            ax.plot(np.arange(1, i + 2), rolling_mean, linewidth=0.8, linestyle='dashed')

        ax.set_xlim(1, N_CHUNKS)
        ax.set_ylim(-0.1, 1.1)

        ax.grid()
        ax.legend(ESTIMATORS_N * 2)

        # TIMES
        ax = fig.add_subplot(grid[1, :])
        ax.set_title("Execution time")
        for s in etimes:
            ax.plot(np.arange(1, i + 2), s[:i + 1], alpha=0.85)

        ax.set_xlim(1, N_CHUNKS)
        ax.set_ylim(etimes.min() - 0.1, etimes.max() + 0.1)

        ax.grid()
        ax.legend(ESTIMATORS_N)

        plt.tight_layout()
        plt.savefig("foo.png")
        plt.close()


if __name__ == '__main__':
    main()
