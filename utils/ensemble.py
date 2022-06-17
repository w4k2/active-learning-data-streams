from sklearn.neighbors import NearestNeighbors
import numpy as np


def get_model_dataset(data, target):
    selected_data = []
    selected_target = []
    for i in range(data.shape[0]):
        num_repeats = np.random.poisson(lam=1.0)
        if num_repeats == 0:
            continue

        for _ in range(num_repeats):
            selected_data.append(data[i])
            selected_target.append(target[i])

    selected_data = np.stack(selected_data, axis=0)
    selected_target = np.stack(selected_target, axis=0)

    return selected_data, selected_target


class Ensemble:
    def __init__(self, models, diversify=False):
        self.models = models
        self.diversify = diversify
        self.seed_data = None
        self.neighbors = NearestNeighbors(n_neighbors=5)

    def fit(self, data, target):
        for model in self.models:
            if self.diversify:
                train_data, train_target = get_model_dataset(data, target)
                model.fit(train_data, train_target)
                self.seed_data = train_data
                self.neighbors.fit(train_data)
            else:
                model.fit(data, target)

    def predict(self, data):
        pred_prob = self.predict_proba(data)
        pred_label = np.argmax(pred_prob, axis=-1)  # TODO should axis be 1 instead of -1 ???
        return pred_label

    def predict_proba(self, data):
        predictions = self.predict_proba_separate(data)
        pred_avrg = np.mean(predictions, axis=0)
        return pred_avrg

    def predict_proba_separate(self, data):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(data)
            predictions.append(pred)
        predictions = np.stack(predictions, axis=0)
        return predictions

    def partial_fit(self, data, target, poisson_lambda=1.0):
        for model in self.models:
            if self.diversify:
                num_repeats = np.random.poisson(lam=poisson_lambda)
                num_repeats = min(num_repeats, 4)  # TODO should we realy use this?
                if num_repeats > 0:
                    target = np.ravel(target)
                    # idx = self.neighbors.kneighbors(data, n_neighbors=num_repeats, return_distance=False)[0]
                    # nearest = self.seed_data[idx]
                    # for neighbor in nearest:
                    #     sample = self._augument_sample(data, neighbor)
                    #     model.partial_fit(sample, target)
                    model.partial_fit(data, target)
            else:
                model.partial_fit(data, target)

    def _augument_sample(self, sample, neighbor):
        vec = neighbor - sample
        norm = np.linalg.norm(vec)
        vec_normalized = vec / norm
        new_len = np.random.rand() * norm
        new_sample = sample + vec_normalized * new_len
        return new_sample
