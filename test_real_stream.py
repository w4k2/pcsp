import numpy as np
import matplotlib.pyplot as plt

from real_stream import RealStreamGenerator
from gif import pyplotGIF

DATASET = 'data/cse/powersupply.arff'

def main():
    stream = RealStreamGenerator(DATASET, chunk_size=100)
    gif = pyplotGIF()

    x_lim = (np.min(stream.X[:, 0]) - 0.1, np.max(stream.X[:, 0]) + 0.1)
    y_lim = (np.min(stream.X[:, 1]) - 0.1, np.max(stream.X[:, 1]) + 0.1)

    for i, (X, y) in enumerate(stream.generate()):
        plt.title(f"{DATASET.split('/')[-1].split('.')[0]} - {[i]}")
        plt.scatter(*X.T, c=y)
        plt.xlim(*x_lim)
        plt.ylim(*y_lim)
        gif.add_frame()
        plt.close()

    print("Exporting gif...")
    gif.export("animame/test_real_stream.gif")

if __name__ == '__main__':
    main()
