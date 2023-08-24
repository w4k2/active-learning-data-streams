import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_classes = len(torch.unique(Y_train))
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.poisson_lambdas = torch.ones_like(Y_train)
        self.is_initialized = False

    def __next__(self):
        if not self.is_initialized:
            raise ValueError("Have to call initialize_labels before iterating")
        
        _, unlabeled_data = self.get_unlabeled_data()

        for i in range(len(unlabeled_data)):
            x, y, idx = unlabeled_data[i]
            yield x, y, idx

    def __iter__(self):
        return next(self)
            
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        self.is_initialized = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler, use_validation_set):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    data = get_data(raw_train, raw_test, handler, use_validation_set)
    return data

def get_FashionMNIST(handler, use_validation_set):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    data = get_data(raw_train, raw_test, handler, use_validation_set)
    return data

def get_SVHN(handler, use_validation_set):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    data = get_data(data_train, data_test, handler, use_validation_set)
    return data

def get_CIFAR10(handler, use_validation_set):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    data = get_data(data_train, data_test, handler, use_validation_set)
    return data

def get_data(raw_train, raw_test, handler, use_validation_set):
    train_data = raw_train.data[:40000]
    train_targets = raw_train.targets[:40000]
    if use_validation_set:
        test_data = train_data[30000:]
        test_targets = train_targets[30000:]
        train_data = train_data[:30000]
        train_targets = train_targets[:30000]
    else:
        test_data = raw_test.data[:40000]
        test_targets = raw_test.targets[:40000]
    data = Data(train_data, train_targets, test_data, test_targets, handler)
    return data