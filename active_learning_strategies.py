import scipy.stats
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


class VoteEntropy(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def request_label(self, obj, current_budget, budget):
        pred_separate = self.model.predict_proba_separate(obj)
        num_classifiers = pred_separate.shape[0]
        idx = np.argmax(pred_separate, axis=2).reshape(num_classifiers, 1, 1)
        pred_probs = np.take_along_axis(pred_separate, idx, axis=2)
        vote_entropy = scipy.stats.entropy(pred_probs.flatten())
        if vote_entropy > self.threshold:
            return True
        else:
            return False
