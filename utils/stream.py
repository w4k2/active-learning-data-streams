from operator import delitem
import sklearn.neighbors
import numpy as np


class Stream:
    def __init__(self, X, y) -> None:
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.len = len(self.X)

        self.density_estimation = sklearn.neighbors.KernelDensity(bandwidth=0.1)
        self.density_estimation.fit(X)

    def __next__(self):
        while len(self.X) > 0:
            sample = self.density_estimation.sample(n_samples=1)
            nearest_idx = self.nearest_neighbor(sample)
            nearest_X = self.X[nearest_idx]
            nearest_y = self.y[nearest_idx]
            self.X = np.delete(self.X, nearest_idx, axis=0)
            self.y = np.delete(self.y, nearest_idx, axis=0)
            yield nearest_X, nearest_y

    def nearest_neighbor(self, sample):
        diff = self.X - sample
        square_diff = np.square(diff)
        dist = np.sqrt(np.sum(square_diff, axis=1))
        nearest_idx = np.argmin(dist)
        return nearest_idx

    def __iter__(self):
        return next(self)

    def __len__(self):
        return self.len
