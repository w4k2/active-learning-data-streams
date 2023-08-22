import numpy as np
import torch
import abc
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy(abc.ABC):
    def __init__(self, dataset, net, threshold=None):
        self.dataset = dataset
        self.net = net
        self.threshold = threshold

    @abc.abstractmethod
    def query(self, n):
        pass

    @abc.abstractmethod
    def should_label(self, X, budget):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs
    
    def predict_prob_raw_data(self, x):
        probs = self.net.predict_prob_raw_data(x)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split_raw_data(self, x, n_drop=10):
        probs = self.net.predict_prob_dropout_split_raw_data(x, n_drop=n_drop)
        return probs
        
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

