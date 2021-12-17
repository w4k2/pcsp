import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from collections import OrderedDict

from sources.helpers.datasets import load_npy
from sources.helpers.constraints import make_constraints, const_mat_to_const_list
from sources.streams.chunk_generator import ChunkGenerator

from sources.ckm import COPKMeans, SCOPKMeans, PCKMeans
from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


C_RATIO = 0.01
G = 5
W = 1.5

MAX_CHUNKS = 500
SYNTHETIC_CHUNKS = 200
CHUNK_SAMPLES = 200

datasets = [
    # "kddcup99",
    # "powersupply",
    # "sensor",
    # "airlines",
    # "covtypeNorm",
    # "elecNormNew",
    "rbf",
    # "static_nooverlaping_balanced",
    # "static_overlaping_balanced",
    # "dynamic_overlaping",
    # "dynamic_radius",
    # "dynamic_imbalance",
    # "strlearn_sudden_drift",
    # "strlearn_gradual_drift",
    # "strlearn_incremental_drift",
    # "strlearn_reccurent_drift",
    # "strlearn_dynamic_imbalanced_drift",
    "strlearn_disco_drift",
]

ESTIMATORS = OrderedDict({
    "KM": KMeans,
    "PCK": PCKMeans,
    "COPK": COPKMeans,
    "SCOPK": SCOPKMeans,
    # "SPCK": SPCKMeans,
})

def main():
    for ds_name in datasets:
        print(f"Running {ds_name} ...")
        X, y = load_npy(ds_name)
        n_clusters = len(np.unique(y))

        estimators = [
            e(n_clusters=n_clusters) for e in ESTIMATORS.values()
        ]

        stream = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES)

        scores = np.zeros((len(estimators), stream.n_chunks_))
        etimes = np.zeros((len(estimators), stream.n_chunks_))

        for i, (X_test, y_test) in tqdm(enumerate(stream), total=stream.n_chunks_):
            const_mat = make_constraints(y_test, ratio=C_RATIO, random_state=100, use_matrix=True)

            for j, est in enumerate(estimators):
                if j == 0:
                    est.fit(X_test)
                elif j == 1:
                    ml, cl = const_mat_to_const_list(const_mat)
                    print(len(ml), len(cl))
                    est.fit(X_test, ml=ml, cl=cl)
                else:
                    est.partial_fit(X_test, const_mat)

                y_pred = est.labels_

                scores[j, i] = adjusted_rand_score(y_test, y_pred)
                etimes[j, i] = float(est.n_iter_)

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
                ax.plot(np.arange(1, i + 2), rolling_mean, linewidth=1.6, alpha=0.8)

            ax.set_xlim(1, stream.n_chunks)
            ax.set_ylim(-0.1, 1.1)
            ax.grid()

            ax.legend(list(ESTIMATORS.keys()))

            # TIMES
            ax = fig.add_subplot(grid[1, :])
            ax.set_title("Iterations")
            for s in etimes:
                ax.plot(np.arange(1, i + 2), s[:i + 1], alpha=0.85)

            ax.set_xlim(1, stream.n_chunks)
            ax.set_ylim(etimes.min() - 0.3, etimes.max() + 5)

            ax.grid()

            ax.legend(list(ESTIMATORS.keys()))

            plt.tight_layout()
            plt.savefig(f"{ds_name}.png")
            plt.close()

        np.save(f"_results_npy/{ds_name}_res.npy", scores)

if __name__ == '__main__':
    main()
