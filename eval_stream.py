import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sources.helpers.datasets import load_npy
from sources.helpers.constraints import make_constraints
from sources.streams.chunk_generator import ChunkGenerator

from sources.kmeans import KMeans, PCKMeans, COPKMeans, SCOPKMeans, SPCKMeans, INIT_RANDOM, INIT_NEIGHBORHOOD
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

import multiprocessing

C_RATIOS = [
    0.001,
    0.005,
    0.010,
]

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
    ("COPK", COPKMeans),
    ("PCK", PCKMeans),
]

ONLINE_ESTIMATORS = [
    ("COPK-S", SCOPKMeans),
    ("PCK-S", SPCKMeans),
]

def eval_dataset(ds_name):
    X, y = load_npy(ds_name)
    n_clusters = len(np.unique(y))

    estimators = [
        e(n_clusters=n_clusters, init=INIT_RANDOM, random_state=SEED)
        for e_name, e in ESTIMATORS
    ]

    online_estimators = [
        e(n_clusters=n_clusters, init=INIT_NEIGHBORHOOD, random_state=SEED)
        for e_name, e in ONLINE_ESTIMATORS
    ]

    for ratio in C_RATIOS:
        stream = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES)
        rs = np.random.default_rng(SEED)

        scores = np.zeros((len(estimators) + len(online_estimators), stream.n_chunks_))
        iters = np.zeros((len(estimators) + len(online_estimators), stream.n_chunks_))

        for i, (X_test, y_test) in tqdm(enumerate(stream), total=stream.n_chunks_, desc=ds_name, position=datasets.index(ds_name)):
            const_mat = make_constraints(y_test, ratio=ratio, random_state=rs)

            for j, e in enumerate(estimators):
                if 'const_mat' in e.fit.__code__.co_varnames:
                    e.fit(X_test, const_mat=const_mat)
                else:
                    e.fit(X_test)

                scores[j, i] = adjusted_rand_score(y_test, e.labels_)
                iters[j, i] = float(e.n_iter_)

            for j, e in enumerate(online_estimators):
                if 'const_mat' in e.fit.__code__.co_varnames:
                    e.partial_fit(X_test, const_mat=const_mat)
                else:
                    e.partial_fit(X_test)

                scores[j + len(estimators), i] = adjusted_rand_score(y_test, e.labels_)
                iters[j + len(estimators), i] = float(e.n_iter_)

        fig = plt.figure(figsize=(G * 3 + 1, W * G))
        grid = fig.add_gridspec(2, 3 + 1)

        # SCORES
        ax = fig.add_subplot(grid[0, :])
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_xlabel("Chunks")

        for s in scores:
            rolling_mean = pd.Series(s[:i + 1]).rolling(window=10).mean()
            ax.plot(np.arange(1, i + 2), rolling_mean, linewidth=1.6, alpha=0.8)

        ax.set_prop_cycle(None)

        for s in scores:
            ax.plot(np.arange(1, i + 2), s[:i + 1], linewidth=0.8, linestyle='dashed', alpha=0.4)

        ax.set_xlim(1, stream.n_chunks)
        ax.set_ylim(-0.1, 1.1)
        ax.grid()

        ax.legend([e_name for e_name, e in ESTIMATORS+ ONLINE_ESTIMATORS])

        # TIMES
        ax = fig.add_subplot(grid[1, :])
        ax.set_ylabel("Iterations")
        for s in iters:
            ax.plot(np.arange(1, i + 2), s[:i + 1], alpha=0.85)

        ax.set_xlim(1, stream.n_chunks)
        ax.set_ylim(iters.min() - 0.3, 100 + 0.3)

        ax.grid()

        plt.tight_layout()
        plt.savefig(f"_eval/{ds_name}_C{ratio:.3f}.png")
        plt.close()

        np.save(f"_results/{ds_name}_C{ratio:.3f}_scores.npy", scores)
        np.save(f"_results/{ds_name}_C{ratio:.3f}_iterations.npy", iters)

def main():
    jobs = []
    for ds_name in datasets:
        p = multiprocessing.Process(target=eval_dataset, args=(ds_name,))
        jobs.append(p)
        p.start()

if __name__ == '__main__':
    main()
