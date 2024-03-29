import numpy as np
import pathlib
import pickle


def get_model_dataset(data, target):
    repeats = np.random.poisson(lam=1.0, size=len(target))

    selected_data, selected_target = sample_data(data, target, repeats)

    all_classes = set(np.unique(target))
    selected_classes = set(np.unique(selected_target))
    missing_classes = all_classes - selected_classes
    for c in missing_classes:
        class_idx = np.argwhere(target.flatten() == c).flatten()
        random_idx = np.random.choice(class_idx, 1)
        extra_sample = data[random_idx].reshape(1, -1)
        extra_target = target[random_idx].reshape(1, -1)
        selected_data = np.concatenate([selected_data, extra_sample], axis=0)
        if selected_target.ndim == 1:
            selected_target = np.expand_dims(selected_target, axis=1)
        selected_target = np.concatenate([selected_target, extra_target], axis=0)

    return selected_data, selected_target


def sample_data(data, target, repeats):
    selected_data = []
    selected_target = []
    for i, num_repeats in enumerate(repeats):
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

    def fit(self, data, target):
        for model in self.models:
            train_data, train_target = get_model_dataset(data, target)
            train_target = np.ravel(train_target)
            model.fit(train_data, train_target)
            self.seed_data = train_data

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

    def partial_fit(self, data, target, poisson_lambdas=None):
        for model in self.models:
            if self.diversify:
                num_repeats = np.random.poisson(lam=poisson_lambdas)
                num_repeats = np.minimum(num_repeats, 4)
                model_data, model_target = sample_data(data, target, num_repeats)
                model_target = np.ravel(model_target)
                model.partial_fit(model_data, model_target)
            else:
                model.partial_fit(data, target)

    def save(self, path: pathlib.Path):
        if not path.exists():
            raise ValueError("Path must exist")

        for i, model in enumerate(self.models):
            with open(path / f'model_{i}.sav', 'wb') as f:
                pickle.dump(model, f)
        with open(path / f'model_diversify.sav', 'wb') as f:
            pickle.dump(self.diversify, f)

    def load(self, path: pathlib.Path):
        if not path.exists():
            raise ValueError("Path must exist")

        self.models = []
        for i in range(9):  # TODO instead of hardcode list all models in directory
            with open(path / f'model_{i}.sav', 'rb') as f:
                model = pickle.load(f)
                self.models.append(model)
        with open(path / f'model_diversify.sav', 'rb') as f:
            self.diversify = pickle.load(f)


if __name__ == '__main__':
    # test is model saving works
    from main import *
    seed_everything(42)

    models = [MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=0.001, max_iter=5000, beta_1=0.9) for _ in range(9)]
    model = utils.ensemble.Ensemble(models, diversify=False)
    data_X = np.random.rand(100, 5)
    data_y = np.random.randint(0, 2, size=(100,))
    model.fit(data_X, data_y)
    test_data = data_X[:5]
    print(model.predict(test_data))
    print(model.predict_proba(test_data))
    import pathlib
    import os
    p = pathlib.Path('./model_test')
    os.makedirs(p, exist_ok=True)
    model.save(p)

    model1 = utils.ensemble.Ensemble([], diversify=False)
    model1.load(p)
    print(model1.predict(test_data))
    print(model1.predict_proba(test_data))
