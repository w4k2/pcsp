import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from stationary_stream import FoldingStationaryStreamGenerator
from stationary_stream import RandomStationaryStreamGenerator
from gif import pyplotGIF


def main():
    X, y = make_classification(n_samples=10000, n_informative=2, n_redundant=0, n_features=2)
    stream = FoldingStationaryStreamGenerator(X, y, n_chunks=100)
    gif = pyplotGIF()

    x_lim = (np.min(stream.X[:, 0]) - 0.1, np.max(stream.X[:, 0]) + 0.1)
    y_lim = (np.min(stream.X[:, 1]) - 0.1, np.max(stream.X[:, 1]) + 0.1)

    for i, (X, y) in enumerate(stream.generate()):
        plt.title(f"Synthetic - {[i]}")
        plt.scatter(*X.T, c=y)
        plt.xlim(*x_lim)
        plt.ylim(*y_lim)
        gif.add_frame()
        plt.close()

    print("Exporting gif...")
    gif.export("animame/test_stationary_stream.gif")


if __name__ == '__main__':
    main()
