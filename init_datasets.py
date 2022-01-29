import matplotlib.pyplot as plt
import strlearn as sl

from sources.streams.moving_blobs import make_moving_blobs
from sources.streams.chunk_generator import ChunkGenerator
from sources.helpers.datasets import *
from sources.helpers.animation import FrameAnimaton

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

MAX_CHUNKS = 200
SYNTHETIC_CHUNKS = MAX_CHUNKS
CHUNK_SAMPLES = 200
MAX_SAMPLES = MAX_CHUNKS * CHUNK_SAMPLES

plt.set_cmap('jet')

def strlearn_gen(stream):
    X, y = [], []

    for i in range(stream.n_chunks):
        X_, y_ = stream.get_chunk()
        X.append(X_)
        y.append(y_)

    return np.concatenate(X), np.concatenate(y)

def prepare_gif(X, y, ds_name):
    print(f"Preparing gif for {ds_name} ...")
    gif = FrameAnimaton()

    X = StandardScaler().fit_transform(X[:MAX_SAMPLES])
    n_clusters = len(np.unique(y[:MAX_SAMPLES]))

    if X.shape[1] > 2:
        print(f"ndim = {X.shape[1]} > 2 - reducing to 2")
        X = PCA(n_components=2).fit_transform(X, y)

    lims = (min(X[:, 0].min(), X[:, 1].min()) * 1.02, max(X[:, 0].max(), X[:, 1].max()) * 1.02)

    s = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES, n_chunks=MAX_CHUNKS)

    for i, (X_, y_) in tqdm(enumerate(s), total=MAX_CHUNKS):
        fig = plt.figure(figsize=(4, 8))
        fig.suptitle(f"{ds_name}")

        grid = fig.add_gridspec(2, 1)

        ax = fig.add_subplot(grid[0, :])
        ax.set_title(f"Chunk: {i}")
        ax.scatter(*X_.T, c=y_, s=5)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid()

        ax = fig.add_subplot(grid[1, :])
        ax.set_title(f"Class distribution")
        bc = np.bincount(y_, minlength=n_clusters) / float(CHUNK_SAMPLES)
        ax.imshow(bc[:, np.newaxis], interpolation='none', vmin=0, vmax=1, cmap='gray_r', aspect='auto', origin='lower')

        plt.tight_layout()
        gif.add_frame()
        plt.close(fig)

    gif.export(f"{ds_name}.gif")

real_datasets = [
        'data/cse/kddcup99.arff',
        'data/cse/powersupply.arff',
        'data/cse/sensor.arff',
        'data/moa/airlines.arff',
        'data/moa/covtypeNorm.arff',
        'data/moa/elecNormNew.arff',
        'data/moa/rbf.arff',
]

concept_kwargs = {
    "n_chunks": SYNTHETIC_CHUNKS,
    "chunk_size": CHUNK_SAMPLES,
    "n_classes": 2,
    "random_state": 200,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_repeated": 0,
}

synthetic_datasets = [
    ('dynamic_overlaping', lambda: make_moving_blobs(
        centers=[
            ((1, 1), (-1, -1)),
            ((-1, -1), (1, 1)),
        ],
        radius=[
            (1, 1),
            (1, 1),
        ],
        weights=[
            (1, 1),
            (1, 1),
        ],
        chunk_size=CHUNK_SAMPLES,
        n_chunks=SYNTHETIC_CHUNKS)
    ),
    ('dynamic_imbalance', lambda: make_moving_blobs(
        centers=[
            ((0, 0), (0, 0)),
            ((-1, -1), (-1, -1)),
        ],
        radius=[
            (1, 1),
            (1, 1),
        ],
        weights=[
            (1, 10),
            (1, 1),
        ],
        chunk_size=CHUNK_SAMPLES,
        n_chunks=SYNTHETIC_CHUNKS)
    ),
    ('strlearn_sudden_drift', lambda: strlearn_gen(sl.streams.StreamGenerator(
            **concept_kwargs, n_drifts=1
        ))
    ),
    ('strlearn_gradual_drift', lambda: strlearn_gen(sl.streams.StreamGenerator(
            **concept_kwargs, n_drifts=1, concept_sigmoid_spacing=5
        ))
    ),
    ('strlearn_static_imbalance', lambda: strlearn_gen(sl.streams.StreamGenerator(
        **concept_kwargs, weights=(0.1, 0.9)
        ))
    ),
    ('strlearn_dynamic_imbalance', lambda: strlearn_gen(sl.streams.StreamGenerator(
        **concept_kwargs, weights=(2, 5, 0.9)
        ))
    ),
]

def main():
    for ds in real_datasets:
        ds_name = ds.split('.')[0].split('/')[-1]

        X_, y_ = load_arff_dataset(ds)
        # limit to chunks on prepare
        X, y = X_[:MAX_SAMPLES], y_[:MAX_SAMPLES]

        print(f"{ds_name} samples:", len(X_))
        print(f"{ds_name} selected:", len(X))

        X = prepare_X(X)
        y = prepare_y(y)
        save_npy(X, y, ds_name)
#        prepare_gif(X, y, ds_name)

    for ds_name, callback in synthetic_datasets:
        X, y = callback()
        print(f"{ds_name} samples:", len(X))
        save_npy(X, y, ds_name)
#        prepare_gif(X, y, ds_name)


if __name__ == '__main__':
    main()
