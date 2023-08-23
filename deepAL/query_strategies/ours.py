import torch
import copy
import math
import numpy as np
import collections
from .strategy import Strategy
from torch.utils.data import Dataset

class SelfLabelingSelectiveSampling(Strategy):
    def __init__(self, dataset, net, threshold, initial_dataset_size, num_models=5):
        super().__init__(dataset, net, threshold)
        self.networks = [copy.deepcopy(self.net) for _ in range(num_models)]
        del self.net
        self.num_classes = dataset.n_classes

        self.num_agree = math.ceil(len(self.networks) / 2)
        self.last_predictions = collections.deque([], maxlen=500)

        self.was_trained = False

    def update(self, pos_idxs, neg_idxs=None):
        # if self.dataset.labeled_idxs[pos_idxs] == True:
        #     raise ValueError
        super().update(pos_idxs, neg_idxs)

    def update_self_labeling(self, pos_idxs, new_labels, poisson_lambda):
        # if self.dataset.labeled_idxs[pos_idxs] == True:
        #     raise ValueError
        self.dataset.labeled_idxs[pos_idxs] = True
        self.dataset.Y_train[pos_idxs] = new_labels
        self.dataset.poisson_lambdas[pos_idxs] = poisson_lambda

    def predict_prob_raw_data(self, x):
        probs = [net.predict_prob_raw_data(x) for net in self.networks]
        probs = torch.mean(torch.stack(probs), dim=0, keepdim=False)
        return probs
    
    def train(self):
        for net in self.networks:
            _, labeled_data = self.dataset.get_labeled_data()
            labeled_idxs = np.arange(self.dataset.n_pool)[self.dataset.labeled_idxs]
            lambdas = self.dataset.poisson_lambdas[labeled_idxs]

            data = labeled_data.X
            target = labeled_data.Y
            original_type = type(data)
            if self.was_trained:
                data, target = self._sample_data(data, target, lambdas)
            else:
                data, target = self._get_initial_dataset(data, target, lambdas)
            if original_type == torch.Tensor:
                data = torch.from_numpy(data)
                target = torch.from_numpy(target)
            labeled_data.X = data
            labeled_data.Y = target
            net.train(labeled_data)
        self.was_trained = True

    def _get_initial_dataset(self, data, target, lambdas):
        selected_data, selected_target = self._sample_data(data, target, lambdas)

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
    
    def _sample_data(self, data, target, lambdas):
        assert len(target) == len(lambdas), f'target and lambdas size mismatch, got: {len(target)} and {len(lambdas)}'
        repeats = np.random.poisson(lam=lambdas)
        repeats = np.minimum(repeats, 4)

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

    def predict(self, data):
        predictions = list()
        for net in self.networks:
            preds = net.predict_prob(data)
            predictions.append(preds)
        predictions = torch.stack(predictions)
        predictions = torch.mean(predictions, dim=0)
        predictions = predictions.max(1)[1]
        return predictions

    def query(self, n):
        pass

    def should_label(self, X, budget):
        probs = [net.predict_prob_raw_data(X) for net in self.networks]
        probs = torch.stack(probs, dim=0)
        supports = probs.numpy()
        predictions = np.argmax(supports, axis=2)

        confident_supports = []
        confident_preds = []
        for supp, pred in zip(supports, predictions):
            if np.max(supp) > self.threshold:
                max_supp = np.max(supp)
                confident_supports.append(max_supp)
                confident_preds.append(pred)

        if len(confident_supports) > self.num_agree and all(pred == confident_preds[0] for pred in confident_preds):
            return False
        else:
            return True

    def use_self_labeling(self, x, current_budget):
        probs = [net.predict_prob_raw_data(x) for net in self.networks]
        probs = torch.stack(probs, dim=0)
        supports = probs.numpy()
        predictions = np.argmax(supports, axis=2)

        confident_supports = []
        confident_preds = []
        confident_models_idxs = set()
        for i, (supp, pred) in enumerate(zip(supports, predictions)):
            if np.max(supp) > self.threshold:
                max_supp = np.max(supp)
                confident_supports.append(max_supp)
                confident_preds.append(pred)
                confident_models_idxs.add(i)

        if len(confident_supports) > self.num_agree and all(pred == confident_preds[0] for pred in confident_preds):
            max_supp = max(confident_supports)
            if current_budget > 0:
                poisson_lambda = max_supp / self.threshold
            else:
                poisson_lambda = abs(self.threshold - max_supp) / self.threshold
            label = confident_preds[0]

            use_selflabeling = True
            if len(self.last_predictions) >= min(self.last_predictions.maxlen, 30):
                current_dist = np.zeros(shape=(self.num_classes,), dtype=int)
                class_label, class_count = np.unique(
                    list(self.last_predictions), return_counts=True)
                for i, count in zip(class_label, class_count):
                    current_dist[i] = count
                current_dist = current_dist / len(self.last_predictions)
                delta_p = current_dist[label] - (1.0 / self.num_classes)
                if delta_p <= 0:
                    use_selflabeling = True
                else:
                    use_selflabeling = False

            if use_selflabeling:
                self.last_predictions.append(int(label))

            train_models = [False for _ in range(len(supports))]
            for idx in range(len(supports)):
                if not idx in confident_models_idxs:
                    train_models[idx] = True

            return use_selflabeling, label, poisson_lambda

        return False, None, 1.0