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

    def fit(self, data, target):
        for model in self.models:
            if self.diversify:
                train_data, train_target = get_model_dataset(data, target)
                model.fit(train_data, train_target)
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
                num_repeats = min(num_repeats, 4)  # TODO should we realy use this
                target = np.ravel(target)
                for _ in range(num_repeats):
                    model.partial_fit(data, target)
            else:
                model.partial_fit(data, target)
