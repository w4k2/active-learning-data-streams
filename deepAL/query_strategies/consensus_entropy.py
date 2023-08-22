import torch
import copy
from .strategy import Strategy

class ConsensusEntropy(Strategy):
    def __init__(self, dataset, net, threshold, num_models=5):
        super().__init__(dataset, net, threshold)
        self.networks = [copy.deepcopy(self.net) for _ in range(num_models)]
        del self.net

    def predict_prob_raw_data(self, x):
        probs = [net.predict_prob_raw_data(x) for net in self.networks]
        probs = torch.mean(torch.stack(probs), dim=0, keepdim=False)
        return probs
    
    def train(self):
        _, labeled_data = self.dataset.get_labeled_data()
        for net in self.networks:
            net.train(labeled_data)

    def predict(self, data):
        predictions = list()
        for net in self.networks:
            preds = net.predict_prob(data)
            predictions.append(preds)
        predictions = torch.stack(predictions)
        predictions = torch.mean(predictions, dim=0)
        predictions = predictions.max(1)[1]
        return predictions

    def query(self, n):
        pass

    def should_label(self, X, budget):
        probs = self.predict_prob_raw_data(X)
        log_probs = torch.log(probs)
        uncertainty = (-probs*log_probs).sum(1)
        return uncertainty > self.threshold
    