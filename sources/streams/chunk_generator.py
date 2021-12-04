import numpy as np


class GeneratorBase:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.active_indices_ = []

    def __iter__(self):
        return self

    def __next__(self):
        self.advance()

        if len(self.active_indices_) == 0:
            raise StopIteration

        return self.X[self.active_indices_], self.y[self.active_indices_]

    def advance(self):
        raise NotImplementedError

    def get_current_indices(self):
        return self.active_indices_


class ChunkGenerator(GeneratorBase):
    def __init__(self, X, y, chunk_size=100, n_chunks=None):
        super(ChunkGenerator, self).__init__(X, y)

        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

        max_chunks = len(X) // chunk_size
        self.n_chunks_ = max_chunks if self.n_chunks is None else min(max_chunks, n_chunks)
        self.chunk_ = 0

    def advance(self):
        if self.chunk_ >= self.n_chunks_:
            self.active_indices_ = []
            return

        start = self.chunk_ * self.chunk_size
        self.active_indices_ = np.arange(start, start + self.chunk_size)
        self.chunk_ += 1


class RandomChunkGenerator(GeneratorBase):
    def __init__(self, X, y, chunk_size=100, n_chunks=None, random_state=None):
        super(RandomChunkGenerator, self).__init__(X, y)

        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.random_state = random_state

        self.rng_ = np.random.default_rng(self.random_state)
        self.chunk_ = 0

    def advance(self):
        if self.n_chunks is not None and self.chunk_ >= self.n_chunks:
            self.active_indices_ = []
            return

        self.active_indices_ = self.rng_.choice(np.arange(len(self.X)), size=self.chunk_size, replace=False)
        self.chunk_ += 1


if __name__ == '__main__':
    N_SAMPLES = 20
    X = np.arange(N_SAMPLES)
    y = np.ones(N_SAMPLES)
    s = RandomChunkGenerator(X, y, chunk_size=10, n_chunks=None, random_state=10)

    print("Manual")

    print(s.get_current_indices())

    s.advance()
    print(s.get_current_indices())

    s.advance()
    print(s.get_current_indices())

    print("Loop")

    s = RandomChunkGenerator(X, y, chunk_size=10, n_chunks=1, random_state=10)
    for X, y in s:
        print(s.active_indices_)
