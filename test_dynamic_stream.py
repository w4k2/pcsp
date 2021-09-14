import numpy as np
import matplotlib.pyplot as plt
from dynamic_stream import RandomDynamicStreamGenerator
from sklearn.datasets import make_circles, make_blobs, make_moons

def main():
    concepts = [
        make_circles(n_samples=10000, noise=0.2),
        make_circles(n_samples=10000),
    ]

    drifts = [
        np.linspace(1, 0, 100),
        np.linspace(0, 1, 100),
    ]

    stream = RandomDynamicStreamGenerator(concepts, drifts, n_chunks=100, chunk_size=1000)

    c = 0
    for X, y in stream.generate():
        print(np.unique(y, return_counts=True))
        plt.scatter(*X.T, c=y)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.savefig(f"gif/{c}.png")
        plt.clf()
        c += 1

if __name__ == '__main__':
    main()
