import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, net, threshold):
        super(LeastConfidence, self).__init__(dataset, net, threshold)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

    def should_label(self, X, budget):
        probs = self.predict_prob_raw_data(X)
        confidence = probs.max(1)[0]
        return confidence < self.threshold