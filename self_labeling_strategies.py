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
        self.prediction_threshold = prediction_threshold

        self.last_predictions = collections.deque([], maxlen=500)
        self.use_selflabeling = True

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

        if len(confident_supports) > 0 and all(pred == confident_preds[0] for pred in confident_preds):
            return False
        else:
            return True

    def use_self_labeling(self, obj, current_budget, budget):
        supports = self.model.predict_proba_separate(obj)
        predictions = np.argmax(supports, axis=2)

        confident_supports = []
        confident_preds = []
        for supp, pred in zip(supports, predictions):
            if np.max(supp) > self.prediction_threshold:
                max_supp = np.max(supp)
                confident_supports.append(max_supp)
                confident_preds.append(pred)

        if len(confident_supports) > 0 and all(pred == confident_preds[0] for pred in confident_preds):
            max_supp = max(confident_supports)
            if current_budget > 0:
                poisson_lambda = max_supp / self.prediction_threshold
            else:
                poisson_lambda = abs(self.prediction_threshold - max_supp) / self.prediction_threshold
            label = confident_preds[0]
            label = np.expand_dims(label, 0)

            if len(self.last_predictions) >= min(self.last_predictions.maxlen, 30):
                _, current_dist = np.unique(
                    list(self.last_predictions), return_counts=True)
                current_dist = current_dist / len(self.last_predictions)
                delta_p = current_dist[label] - (1.0 / self.num_classes)
                if delta_p <= 0:
                    self.use_selflabeling = True
                else:
                    self.use_selflabeling = False

            if self.use_selflabeling:
                self.last_predictions.append(int(label))

            return self.use_selflabeling, label, {'poisson_lambda': poisson_lambda}

        return False, None, {}
