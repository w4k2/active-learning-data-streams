import numpy as np
import random
import os
import matplotlib.pyplot as plt
import mkl
import collections

from sklearn.neural_network import MLPClassifier


def main():
    mkl.set_num_threads(4)
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    X_train, y_train = get_dataset1(300)
    X_stream, y_stream = get_dataset1(3000)
    plot_data(X_train, y_train, (2, 3, 1))
    class_percentage = experiment(X_train, y_train, X_stream, y_stream)
    plot_percentage(class_percentage, (2, 3, 2))
    class_percentage = experiment_ballanced(X_train, y_train, X_stream, y_stream)
    plot_percentage(class_percentage, (2, 3, 3))

    X_train, y_train = get_dataset2(240, 60)
    X_stream, y_stream = get_dataset2(2400, 600)
    plot_data(X_train, y_train, (2, 3, 4))
    class_percentage = experiment(X_train, y_train, X_stream, y_stream)
    plot_percentage(class_percentage, (2, 3, 5))
    class_percentage = experiment_ballanced(X_train, y_train, X_stream, y_stream)
    plot_percentage(class_percentage, (2, 3, 6))

    plt.show()


def get_dataset1(dataset_size):
    size = dataset_size // 3
    class1_data = np.random.multivariate_normal([-2, 1], [[5, 2], [2, 1]], size=size)
    class2_data = np.random.multivariate_normal([2, 1], [[5, -2], [-2, 2]], size=size)
    class3_data = np.random.multivariate_normal([0, -2.0], [[1.5, 0], [0, 1.5]], size=size)
    X = np.concatenate((class1_data, class2_data, class3_data), axis=0)
    y = np.concatenate((np.zeros(size), np.ones(size), np.ones(size)*2), axis=0)
    idx = np.random.permutation(3*size).flatten()
    X = X[idx]
    y = y[idx]
    return X, y


def get_dataset2(size1, size2):
    class1_data = np.random.multivariate_normal([-1.5, 0], [[2, 1], [1, 2]], size=size1)
    class2_data = np.random.multivariate_normal([1.5, 0], [[1, 0], [0, 1]], size=size2)
    X = np.concatenate((class1_data, class2_data), axis=0)
    y = np.concatenate((np.zeros(size1), np.ones(size2)), axis=0)
    idx = np.random.permutation(size1+size2).flatten()
    X = X[idx]
    y = y[idx]
    return X, y


def plot_data(X, y, subplot_idx):
    plt.subplot(*subplot_idx)
    classes = np.unique(y)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for c in classes:
        x1, y1 = X[np.argwhere(y == c).flatten()].T
        plt.plot(x1, y1, '.', color=colors[int(c)], label=f"class {int(c)+1}")

    plt.legend()


def plot_percentage(class_percentage, subplot_idx):
    plt.subplot(*subplot_idx)
    class_percentage = np.array(class_percentage)
    cumulative = np.cumsum(class_percentage, axis=1)
    num_classes = class_percentage.shape[1]

    x = list(range(len(cumulative)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(num_classes-1, -1, -1):
        plt.fill_between(x, np.zeros_like(x), cumulative[:, i], color=colors[i])

    plt.xlabel('iterations')
    plt.ylabel('training set class percentage')
    plt.xlim(0, len(class_percentage))
    plt.ylim(0, 1)


def experiment(X_train, y_train, X_stream, y_stream):
    X_train_exp = np.copy(X_train)
    y_train_exp = np.copy(y_train)
    model = MLPClassifier(hidden_layer_sizes=(5, ), learning_rate_init=0.01, max_iter=100)
    model.fit(X_train_exp, y_train_exp)

    class_percentage = [[len(np.argwhere(y_train == i)) / len(y_train) for i in range(3)]]

    for sample, target in zip(X_stream, y_stream):
        sample = np.expand_dims(sample, axis=0)
        target = np.expand_dims(target, axis=0)

        y_pred = model.predict_proba(sample)
        label = np.argmax(y_pred, axis=1)
        if y_pred.max() < 0.7:
            X_train_exp = np.concatenate((X_train_exp, sample), axis=0)
            y_train_exp = np.concatenate((y_train_exp, target), axis=0)
            model.partial_fit(X_train_exp, y_train_exp)
        elif y_pred.max() > 0.95:
            X_train_exp = np.concatenate((X_train_exp, sample), axis=0)
            y_train_exp = np.concatenate((y_train_exp, label), axis=0)
            model.partial_fit(X_train_exp, y_train_exp)

            current_percentage = [len(np.argwhere(y_train_exp == i)) / len(y_train_exp) for i in range(3)]
            class_percentage.append(current_percentage)
    return class_percentage


def experiment_ballanced(X_train, y_train, X_stream, y_stream):
    model = MLPClassifier(hidden_layer_sizes=(5, ), learning_rate_init=0.01, max_iter=100)
    model.fit(X_train, y_train)
    num_classes = len(np.unique(y_train).flatten())

    last_predictions = collections.deque([], maxlen=50)
    class_percentage = [[len(np.argwhere(y_train == i)) / len(y_train) for i in range(3)]]

    for sample, target in zip(X_stream, y_stream):
        sample = np.expand_dims(sample, axis=0)
        target = np.expand_dims(target, axis=0)

        y_pred = model.predict_proba(sample)
        label = np.argmax(y_pred, axis=1)
        if y_pred.max() < 0.7:
            X_train = np.concatenate((X_train, sample), axis=0)
            y_train = np.concatenate((y_train, target), axis=0)
            last_predictions.append(int(target))
        elif y_pred.max() > 0.9:
            use_selflabeling = True
            if len(last_predictions) >= 30:
                current_dist = np.zeros(shape=(num_classes,), dtype=int)
                class_label, class_count = np.unique(list(last_predictions), return_counts=True)
                for i, count in zip(class_label, class_count):
                    current_dist[i] = count
                current_dist = current_dist / len(last_predictions)
                delta_p = current_dist[label] - (1.0 / num_classes)
                if delta_p <= 0:
                    use_selflabeling = True
                else:
                    use_selflabeling = False

            if use_selflabeling:
                last_predictions.append(int(label))
                X_train = np.concatenate((X_train, sample), axis=0)
                y_train = np.concatenate((y_train, label), axis=0)
                model.partial_fit(X_train, y_train)

                current_percentage = [len(np.argwhere(y_train == i)) / len(y_train) for i in range(3)]
                class_percentage.append(current_percentage)
    return class_percentage


if __name__ == '__main__':
    main()
