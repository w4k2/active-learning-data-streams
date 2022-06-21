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


class VariableUncertainty(ActiveLearningStrategy):
    def __init__(self, model, threshold, adjusting_step=0.01):
        super().__init__(model)
        self.threshold = threshold
        self.labeling_cost = 0
        self.adjusting_step = adjusting_step

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba(obj)
        max_prob = np.max(pred_prob, axis=1)[0]
        if max_prob < self.threshold:
            self.labeling_cost += 1
            self.threshold = self.threshold * (1 - self.adjusting_step)
            return True
        else:
            self.threshold = self.threshold * (1 + self.adjusting_step)
            return False


class VariableRandomizedUncertainty(ActiveLearningStrategy):
    def __init__(self, model, threshold, adjusting_step=0.01, radomization_variance=1):
        super().__init__(model)
        self.threshold = threshold
        self.labeling_cost = 0
        self.adjusting_step = adjusting_step
        self.radomization_variance = radomization_variance

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba(obj)
        max_prob = np.max(pred_prob, axis=1)[0]
        eta = np.random.randn() * self.radomization_variance + 1.0
        threshold_randomized = eta * self.threshold
        if max_prob < threshold_randomized:
            self.labeling_cost += 1
            self.threshold = self.threshold * (1 - self.adjusting_step)
            return True
        else:
            self.threshold = self.threshold * (1 + self.adjusting_step)
            return False
