import numpy as np
import matplotlib.pyplot as plt

from .blobs_generator import make_beta_blobs

N_CHUNKS = 100


def make_moving_blobs(centers, radius=None, weights=None, chunk_size=10, n_chunks=10):
    X, y = [], []

    centers_series = np.array([np.linspace(s, e, n_chunks) for s, e in centers]).transpose(1, 0, 2)
    radius_series = np.array([np.linspace(s, e, n_chunks) for s, e in radius]).T
    weights_series = np.array([np.linspace(s, e, n_chunks) for s, e in weights]).T
    # Normalize
    weights_series = weights_series / np.sum(weights_series, axis=1)[:, np.newaxis]

    for chunk_centers, chunk_radius, chunk_weights in zip(centers_series, radius_series, weights_series):
        X_, y_ = make_beta_blobs(chunk_centers, radius=chunk_radius, weights=chunk_weights, n_samples=chunk_size)
        X.append(X_)
        y.append(y_)

    return np.concatenate(X), np.concatenate(y)

def main():
    N_CHUNKS = 5
    CHUNK_SIZE = 100

    X, y = make_moving_blobs(
        centers = [
            ((0, 0), (1, 1)),
            ((0, 0), (-1, -1)),
        ],
        radius=[
            (1, 1),
            (1, 2),
        ],
        weights=[
            (1, 1),
            (10, 1),
        ],
        chunk_size=CHUNK_SIZE,
        n_chunks=N_CHUNKS)

    for _ in range(N_CHUNKS):
        ind = np.arange(_ * CHUNK_SIZE, (_ + 1) * CHUNK_SIZE)
        print(np.unique(y[ind], return_counts=True))
        plt.scatter(*X[ind].T, c=y[ind])
        plt.show()


if __name__ == '__main__':
    main()
