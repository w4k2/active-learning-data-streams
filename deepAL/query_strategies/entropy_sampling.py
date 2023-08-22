import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, net, threshold):
        super(EntropySampling, self).__init__(dataset, net, threshold)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

    def should_label(self, X, budget):
        probs = self.predict_prob_raw_data(X)
        log_probs = torch.log(probs)
        uncertainty = (-probs*log_probs).sum(1)
        return uncertainty > self.threshold