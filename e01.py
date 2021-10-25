import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sources.helpers.datasets import load_npy
from sources.helpers.constraints import make_constraints
from sources.streams.chunk_generator import ChunkGenerator

from sources.ckm.cop_kmeans import COPKMeans
from sources.ckm.pcs_kmeans import PCSKMeans
from sources.ckm.mini_batch.cop_kmeans import MiniBatchCOPKmeans

from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

C_RATIO = 0.01
G = 5
W = 1.5

MAX_CHUNKS = 500
SYNTHETIC_CHUNKS = 200
CHUNK_SAMPLES = 200

datasets = [
    "kddcup99",
    "powersupply",
    "sensor",
    "static_nooverlaping_balanced",
    "static_overlaping_balanced",
    "dynamic_overlaping",
    "dynamic_radius",
    "dynamic_imbalance",
]

ESTIMATORS_N = [
    "PCSKMeans",
    "COPKMeans",
    "MiniBatchCOPKmeans",
]

def main():
    for ds_name in datasets:
        print(f"Running {ds_name} ...")
        X, y = load_npy(ds_name)
        n_clusters = len(np.unique(y))

        estimators = [
            PCSKMeans(n_clusters=n_clusters, random_state=100),
            COPKMeans(n_clusters=n_clusters, random_state=100),
            MiniBatchCOPKmeans(n_clusters=n_clusters, random_state=100),
        ]

        stream = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES)

        scores = np.zeros((len(estimators), stream.n_chunks_))
        etimes = np.zeros((len(estimators), stream.n_chunks_))

        for i, (X_test, y_test) in tqdm(enumerate(stream), total=stream.n_chunks_):
            const_mat = make_constraints(y_test, ratio=C_RATIO, random_state=100)

            for j, est in enumerate(estimators):
                est.partial_fit(X_test, const_mat)
                y_pred = est.labels_

                scores[j, i] = adjusted_rand_score(y_test, y_pred)
                etimes[j, i] = float(est.n_iter_)

        np.save(f"_results_npy/{ds_name}_res.npy", scores)

        fig = plt.figure(figsize=(G * 3 + 1, W * G))
        grid = fig.add_gridspec(2, 3 + 1)

        # SCORES
        ax = fig.add_subplot(grid[0, :])
        ax.set_title("Performance")
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_xlabel("Chunks")

        for s in scores:
            ax.plot(np.arange(1, i + 2), s[:i + 1], linewidth=0.8, linestyle='dashed', alpha=0.4)

        ax.set_prop_cycle(None)

        for s in scores:
            rolling_mean = pd.Series(s[:i + 1]).rolling(window=10).mean()
            ax.plot(np.arange(1, i + 2), rolling_mean, linewidth=1.6)

        ax.set_xlim(1, stream.n_chunks)
        ax.set_ylim(-0.1, 1.1)
        ax.grid()

        ax.legend(ESTIMATORS_N)

        # TIMES
        ax = fig.add_subplot(grid[1, :])
        ax.set_title("Iterations")
        for s in etimes:
            ax.plot(np.arange(1, i + 2), s[:i + 1], alpha=0.85)

        ax.set_xlim(1, stream.n_chunks)
        ax.set_ylim(etimes.min() - 0.3, etimes.max() + 5)

        ax.grid()

        ax.legend(ESTIMATORS_N)

        plt.tight_layout()
        plt.savefig(f"{ds_name}.png")
        plt.close()


if __name__ == '__main__':
    main()
