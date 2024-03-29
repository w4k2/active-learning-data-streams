import math
import abc
import collections
import numpy as np


class SelfLabelingStrategy(abc.ABC):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def request_label(self, obj, current_budget, budget):
        raise NotImplementedError

    @abc.abstractmethod
    def use_self_labeling(self, obj, current_budget, budget):
        raise NotImplementedError


class Ours(SelfLabelingStrategy):
    def __init__(self, model, num_classes, prediction_threshold) -> None:
        super().__init__(model)

        self.num_classes = num_classes
        self.num_agree = math.ceil(len(model.models) / 2)
        self.prediction_threshold = prediction_threshold
        self.last_predictions = collections.deque([], maxlen=500)

    def request_label(self, obj, current_budget, budget):
        supports = self.model.predict_proba_separate(obj)
        predictions = np.argmax(supports, axis=2)

        confident_supports = []
        confident_preds = []
        for supp, pred in zip(supports, predictions):
            if np.max(supp) > self.prediction_threshold:
                max_supp = np.max(supp)
                confident_supports.append(max_supp)
                confident_preds.append(pred)

        if len(confident_supports) > self.num_agree and all(pred == confident_preds[0] for pred in confident_preds):
            return False
        else:
            return True

    def use_self_labeling(self, obj, current_budget, budget):
        supports = self.model.predict_proba_separate(obj)
        predictions = np.argmax(supports, axis=2)

        confident_supports = []
        confident_preds = []
        confident_models_idxs = set()
        for i, (supp, pred) in enumerate(zip(supports, predictions)):
            if np.max(supp) > self.prediction_threshold:
                max_supp = np.max(supp)
                confident_supports.append(max_supp)
                confident_preds.append(pred)
                confident_models_idxs.add(i)

        if len(confident_supports) > self.num_agree and all(pred == confident_preds[0] for pred in confident_preds):
            max_supp = max(confident_supports)
            if current_budget > 0:
                poisson_lambda = max_supp / self.prediction_threshold
            else:
                poisson_lambda = abs(self.prediction_threshold - max_supp) / self.prediction_threshold
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

        return False, None, {}
