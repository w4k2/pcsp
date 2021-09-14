import numpy as np
import matplotlib.pyplot as plt
from stationary_stream import FoldingStationaryStreamGenerator
from stationary_stream import RandomStationaryStreamGenerator
from sklearn.datasets import make_circles, make_blobs, make_moons, make_classification

def main():
    X, y = make_classification(n_samples=1000, n_informative=2, n_redundant=0, n_features=2)

    x_lim = (np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1)
    y_lim = (np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1)

    # stream = FoldingStationaryStreamGenerator(X, y, n_chunks=100)
    stream = RandomStationaryStreamGenerator(X, y, n_chunks=100, chunk_size=100)

    c = 0
    for X, y in stream.generate():
        print(np.unique(y, return_counts=True))
        plt.scatter(*X.T, c=y)
        plt.xlim(*x_lim)
        plt.ylim(*y_lim)
        plt.savefig(f"gif/foo_{c}.png")
        plt.clf()
        c += 1

if __name__ == '__main__':
    main()
