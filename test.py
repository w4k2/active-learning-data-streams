import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

model = MLPClassifier()
data, target = load_iris(return_X_y=True)
# print(data)

model.fit(data, target)

obj = np.expand_dims(data[0], axis=0)
target = np.expand_dims(target[0], axis=0)

weights_before = model.coefs_[0][0]
model.partial_fit(obj, target)
weights_after = model.coefs_[0][0]

print(weights_before)
print(weights_after)  # are the same for some reason ???
