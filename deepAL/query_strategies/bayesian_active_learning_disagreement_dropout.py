import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, net, threshold, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net, threshold)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

    def should_label(self, X, budget):
        probs = self.predict_prob_dropout_split_raw_data(X)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainty = entropy2 - entropy1
        return uncertainty < self.threshold
