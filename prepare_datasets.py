import matplotlib.pyplot as plt

from sources.streams.moving_blobs import make_moving_blobs
from sources.streams.chunk_generator import ChunkGenerator
from sources.helpers.datasets import *
from sources.helpers.animation import FrameAnimaton

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler

MAX_CHUNKS = 500
SYNTHETIC_CHUNKS = 200
CHUNK_SAMPLES = 200
MAX_SAMPLES = MAX_CHUNKS * CHUNK_SAMPLES


def prepare_gif(X, y, ds_name):
    print(f"Preparing gif for {ds_name} ...")
    gif = FrameAnimaton()

    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 2:
        # Neeeeeeds reduuuuction
        X = PCA(n_components=2).fit_transform(X, y)

    # X = StandardScaler().fit_transform(X)
    lims = (min(X[:, 0].min(), X[:, 1].min()) * 1.02, max(X[:, 0].max(), X[:, 1].max()) * 1.02)
    # print(lims)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.set_title(f"Complete")
    ax.scatter(*X.T, c=y, s=3)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    plt.tight_layout()
    plt.savefig(f"{ds_name}.png")
    plt.close(fig)

    s = ChunkGenerator(X, y, chunk_size=CHUNK_SAMPLES, n_chunks=MAX_CHUNKS)

    for i, (X_, y_) in tqdm(enumerate(s), total=MAX_CHUNKS):
        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()

        ax.set_title(f"{i}")
        ax.scatter(*X_.T, c=y_, s=3)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        plt.tight_layout()
        gif.add_frame()
        plt.close(fig)

    gif.export(f"{ds_name}.gif")

real_datasets = [
        'data/cse/kddcup99.arff',
        'data/cse/powersupply.arff',
        'data/cse/sensor.arff',
]

synthetic_datasets = [
    ('static_nooverlaping_balanced', lambda: make_moving_blobs(
        centers=[
            ((1, 1), (1, 1)),
            ((-1, -1), (-1, -1)),
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
    ('static_overlaping_balanced', lambda: make_moving_blobs(
        centers=[
            ((0, 0), (0, 0)),
            ((-1, -1), (-1, -1)),
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
    ('dynamic_radius', lambda: make_moving_blobs(
        centers=[
            ((1, 1), (1, 1)),
            ((-1, -1), (-1, -1)),
        ],
        radius=[
            (1, 3),
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
]

def main():
    for ds in real_datasets:
        ds_name = ds.split('.')[0].split('/')[-1]
        X, y = load_arff_dataset(ds)
        print(f"{ds_name} samples:", len(X))
        # limit to chunks an prepare
        X, y = X[:MAX_SAMPLES], y[:MAX_SAMPLES]
        print(f"{ds_name} selected:", len(X))
        X = prepare_X(X)
        y = prepare_y(y)
        save_npy(X, y, ds_name)
        prepare_gif(X, y, ds_name)

    for ds_name, callback in synthetic_datasets:
        X, y = callback()
        save_npy(X, y, ds_name)
        prepare_gif(X, y, ds_name)


if __name__ == '__main__':
    main()
