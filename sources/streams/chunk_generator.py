import numpy as np

class ChunkGenerator:
    def __init__(self, X, y, chunk_size=100):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size

        self._curr_i = 0
        self._next_i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.advance()

        if self._next_i > len(self.X):
            raise StopIteration

        ind = self.get_current_indices()
        return self.X[ind], self.y[ind]

    def advance(self):
        self._curr_i = self._next_i
        self._next_i += self.chunk_size

    def get_current_indices(self):
        return range(self._curr_i, self._next_i)


class RandomChunkGenerator():
    def __init__(self, X, y, chunk_size=100, random_state=None):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size
        self.random_state = random_state

        self._rng = np.random.default_rng(self.random_state)
        self._curr_ind = None

    def __next__(self):
        self.advance()
        ind = self.get_current_indices()
        return self.X[ind], self.y[ind]

    def __iter__(self):
        return self

    def advance(self):
        self._curr_ind = self._rng.choice(range(len(self.X), size=self.chunk_size, replace=False))

    def get_current_indices(self):
        return self._curr_ind


def main():
    import time
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=10000)

    xlim=(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1)
    ylim=(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1)

    s = ChunkGenerator(X, y)

    for X_, y_ in s:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes()

        ax.scatter(*X_.T, c=y_)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plt.tight_layout()
        plt.savefig("foo.png")
        time.sleep(0.1)
        plt.close()

if __name__ == '__main__':
    main()
