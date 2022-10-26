import scipy.stats
import abc
import numpy as np

import utils.ensemble


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


class ClassificationMargin(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def request_label(self, obj, current_budget, budget):
        pred_probs = self.model.predict_proba(obj)
        partition = np.partition(-pred_probs, 1, axis=1)
        margin = - partition[:, 0] + partition[:, 1]
        return margin < self.threshold


class VoteEntropy(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        assert type(model) == utils.ensemble.Ensemble, "Vote entropy must use ensemble model"
        super().__init__(model)
        self.threshold = threshold
        self.num_classifiers = len(model.models)

    def request_label(self, obj, current_budget, budget):
        pred_separate = self.model.predict_proba_separate(obj)
        idx = np.argmax(pred_separate, axis=2).reshape(self.num_classifiers, 1, 1)
        pred_probs = np.take_along_axis(pred_separate, idx, axis=2)
        vote_entropy = scipy.stats.entropy(pred_probs.flatten())
        return vote_entropy > self.threshold


class ConsensusEntropy(ActiveLearningStrategy):
    def __init__(self, model, threshold, num_classes):
        super().__init__(model)
        self.threshold = threshold
        self.max_entropy = scipy.stats.entropy([1.0/num_classes for _ in range(num_classes)])

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba(obj)
        entropy = scipy.stats.entropy(pred_prob, axis=1) / self.max_entropy
        return entropy > self.threshold


class MaxDisagreement(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba_separate(obj)
        pred_prob_consensus = np.mean(pred_prob, axis=0, keepdims=False)
        classifers_KL_divergence = [scipy.stats.entropy(pred_prob[i], qk=pred_prob_consensus, axis=1) for i in range(len(pred_prob))]
        max_kl_divergence = np.max(classifers_KL_divergence)
        return max_kl_divergence > self.threshold


class MinMargin(ActiveLearningStrategy):
    def __init__(self, model, threshold):
        assert type(model) == utils.ensemble.Ensemble, "Vote entropy must use ensemble model"
        super().__init__(model)
        self.threshold = threshold
        self.num_classifiers = len(model.models)

    def request_label(self, obj, current_budget, budget):
        pred_prob = self.model.predict_proba_separate(obj)  # shape: num_classifiers x num_samples x num_classes

        partition = np.partition(-pred_prob, 1, axis=2)
        margins = - partition[:, :, 0] + partition[:, :, 1]
        m = np.min(margins, axis=0, keepdims=False)
        return m < self.threshold
