import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net, threshold):
        super(RandomSampling, self).__init__(dataset, net, threshold)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)

    def should_label(self, X, budget):
        eta_t = np.random.uniform(0, 1)
        return eta_t < budget