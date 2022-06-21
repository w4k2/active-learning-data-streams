import abc
import numpy as np


class ActiveLearningStrategy(abc.ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def request_label(self, obj, current_budget, budget):
        raise NotImplementedError


class RandomSampling(ActiveLearningStrategy):
    def __init__(self, model):
        super().__init__(model)

    def request_label(self, obj, current_budget, budget):
        eta_t = np.random.uniform(0, 1)
        return eta_t < budget


class FixedUncertainty(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba(obj)
        max_prob = np.max(pred_prob, axis=1)[0]
        return max_prob < self.threshold
