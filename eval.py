import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sources.helpers.datasets import load_npy
from sources.helpers.constraints import make_constraints
from sources.streams.chunk_generator import ChunkGenerator

from sources.kmeans import KMeans, PCKMeans, COPKMeans, SCOPKMeans, INIT_RANDOM
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


C_RATIO = 0.01
G = 5
W = 1.5

MAX_CHUNKS = 500
SYNTHETIC_CHUNKS = 200
CHUNK_SAMPLES = 200
SEED = 100

datasets = [
    'kddcup99',
    'powersupply',
    'sensor',
    'airlines',
    'covtypeNorm',
    'elecNormNew',
    "dynamic_overlaping",
    "dynamic_imbalance",
    "strlearn_sudden_drift",
    "strlearn_gradual_drift",
    "strlearn_static_imbalance",
    "strlearn_dynamic_imbalance",
]

ESTIMATORS = [
    ("KM", KMeans),
    ("PCK", PCKMeans),
    ("COPK", COPKMeans),
    ("SCOPK", SCOPKMeans),
]

def eval_dataset(ds_name):
    X, y = load_npy(ds_name)
    n_clusters = len(np.unique(y))

    estimators = [
        e(n_clusters=n_clusters, init=INIT_RANDOM, random_state=SEED) for e_name, e in ESTIMATORS
    ]

    stream = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES)

    scores = np.zeros((len(estimators), stream.n_chunks_))
    iters = np.zeros((len(estimators), stream.n_chunks_))

    for i, (X_test, y_test) in tqdm(enumerate(stream), total=stream.n_chunks_):
        const_mat = make_constraints(y_test, ratio=C_RATIO, random_state=SEED)

        for j, e in enumerate(estimators):
            if 'const_mat' in e.fit.__code__.co_varnames:
                e.partial_fit(X_test, const_mat=const_mat)
            else:
                e.partial_fit(X_test)

            scores[j, i] = adjusted_rand_score(y_test, e.labels_)
            iters[j, i] = float(e.n_iter_)

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

        ax.legend([e_name for e_name, e in ESTIMATORS])

        # TIMES
        ax = fig.add_subplot(grid[1, :])
        ax.set_ylabel("Iterations")
        for s in iters:
            ax.plot(np.arange(1, i + 2), s[:i + 1], alpha=0.85)

        ax.set_xlim(1, stream.n_chunks)
        ax.set_ylim(iters.min() - 0.3, 100 + 0.3)

        ax.grid()

        ax.legend([e_name for e_name, e in ESTIMATORS])

        plt.tight_layout()
        plt.savefig(f"eval/eval_{ds_name}.png")
        plt.savefig(f"foo.png")
        plt.close()

    np.save(f"_results_npy/{ds_name}_res.npy", scores)

def main():
    for ds_name in datasets:
        print(f"Running {ds_name} ...")
        eval_dataset(ds_name)

if __name__ == '__main__':
    main()
