import numpy as np

from sklearn.neighbors import NearestNeighbors


class Model:
    def __init__(self, base_model):
        self.base_model = base_model
        self.seed_data = None
        self.neighbors = NearestNeighbors(n_neighbors=1)

    def fit(self, data, target):
        self.base_model.fit(data, target)
        self.seed_data = data
        self.neighbors.fit(data)

    def predict(self, data):
        return self.base_model.predict(data)

    def predict_proba(self, data):
        return self.base_model.predict_proba(data)

    def partial_fit(self, data, target):
        target = np.ravel(target)
        idx = self.neighbors.kneighbors(data, return_distance=False)[0]
        nearest = self.seed_data[idx]
        sample = self._augument_sample(data, nearest)
        # sample = data
        self.base_model.partial_fit(sample, target)

    def _augument_sample(self, sample, neighbor):
        vec = neighbor - sample
        norm = np.linalg.norm(vec)
        vec_normalized = vec / norm
        new_len = np.random.rand() * norm * 0.01
        new_sample = sample + vec_normalized * new_len
        return new_sample
