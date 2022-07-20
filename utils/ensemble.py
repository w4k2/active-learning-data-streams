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

    all_classes = set(np.unique(target))
    selected_classes = set(np.unique(selected_target))
    missing_classes = all_classes - selected_classes
    for c in missing_classes:
        class_idx = np.argwhere(target.flatten() == c).flatten()
        random_idx = np.random.choice(class_idx, 1)
        extra_sample = data[random_idx].reshape(1, -1)
        extra_target = target[random_idx].reshape(1, -1)
        selected_data = np.concatenate([selected_data, extra_sample], axis=0)
        selected_target = np.concatenate([selected_target, extra_target], axis=0)

    return selected_data, selected_target


class Ensemble:
    def __init__(self, models, diversify=False):
        self.models = models
        self.diversify = diversify
        self.seed_data = None

    def fit(self, data, target):
        for model in self.models:
            if self.diversify:
                train_data, train_target = get_model_dataset(data, target)
                train_target = np.ravel(train_target)
                model.fit(train_data, train_target)
                self.seed_data = train_data
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
        """
        returns array with shape [num_classifiers, num_samples, num_classes]
        """
        predictions = []
        for model in self.models:
            pred = model.predict_proba(data)
            predictions.append(pred)
        predictions = np.stack(predictions, axis=0)
        return predictions

    def partial_fit(self, data, target, poisson_lambda=1.0, train_models=None):
        if train_models is None:
            train_models = (True for _ in range(len(self.models)))

        for train, model in zip(train_models, self.models):
            if not train:
                continue
            if self.diversify:
                num_repeats = np.random.poisson(lam=poisson_lambda)
                num_repeats = min(num_repeats, 4)
                if num_repeats > 0:
                    target = np.ravel(target)
                    model.partial_fit(data, target)
            else:
                model.partial_fit(data, target)
