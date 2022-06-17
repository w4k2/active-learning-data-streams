from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


class Model:
    def __init__(self, base_model):
        self.base_model = base_model
        self.seed_data = None
        self.neighbors = NearestNeighbors(n_neighbors=1)

    def fit(self, data, target):
        self.base_model.fit(data, target)
        self.seed_data = data
        self.seed_target = target
        self.neighbors.fit(data)

    def predict(self, data):
        return self.base_model.predict(data)

    def predict_proba(self, data):
        return self.base_model.predict_proba(data)

    def partial_fit(self, data, target, visualisation=False):
        target = np.ravel(target)
        # idx = self.neighbors.kneighbors(data, return_distance=False)[0]
        # nearest = self.seed_data[idx]
        # sample = self._augument_sample(data, nearest)
        # if visualisation:
        #     self.tsne_plot(data, target, nearest, sample)

        self.base_model.partial_fit(data, target)

    def tsne_plot(self, data, target, nearest, sample):
        print('label assigned to data = ', target)
        all_data = np.concatenate((self.seed_data, nearest, sample, data), axis=0)
        X_transformed = TSNE(n_components=2).fit_transform(all_data)

        class_1_idx = np.argwhere(self.seed_target == 0).flatten()
        X_transformed_class_1 = X_transformed[class_1_idx]
        plt.scatter(X_transformed_class_1[:-3, 0], X_transformed_class_1[:-3, 1], c='black', marker='o', label='class 0')

        class_2_idx = np.argwhere(self.seed_target == 1).flatten()
        X_transformed_class_2 = X_transformed[class_2_idx]
        plt.scatter(X_transformed_class_2[:-3, 0], X_transformed_class_2[:-3, 1], c='black', marker='v', label='class 1')

        class_3_idx = np.argwhere(self.seed_target == 2).flatten()
        X_transformed_class_3 = X_transformed[class_3_idx]
        plt.scatter(X_transformed_class_3[:-3, 0], X_transformed_class_3[:-3, 1], c='black', marker='s', label='class 2')

        plt.scatter(X_transformed[-3, 0], X_transformed[-3, 1], c='g', marker='.', label='nn')
        plt.scatter(X_transformed[-2, 0], X_transformed[-2, 1], c='r', marker='.', label='aug')
        plt.scatter(X_transformed[-1, 0], X_transformed[-1, 1], c='b', marker='.', label='data')

        plt.legend()
        plt.show()
        exit()

    def _augument_sample(self, sample, neighbor):
        vec = neighbor - sample
        norm = np.linalg.norm(vec)
        vec_normalized = vec / norm
        new_len = np.random.rand() * norm * 0.1
        new_sample = sample + vec_normalized * new_len
        return new_sample
