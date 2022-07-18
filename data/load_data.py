import numpy as np
import csv
import sklearn.model_selection


def get_data(dataset_name, seed_size, test_size, random_seed):
    X, y, num_classes = load_dataset(dataset_name)

    X, seed_data, y, seed_target = sklearn.model_selection.train_test_split(X, y, test_size=seed_size, random_state=random_seed)
    X_train, test_data, y_train, test_target = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_seed)
    stream = list(zip(X_train, y_train))
    return seed_data, seed_target, test_data, test_target, stream, num_classes


def load_dataset(dataset_name):
    if dataset_name == 'accelerometer':
        data = []
        labels = []
        with open('data/accelerometer.csv', 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                labels.append(float(line[0]))
                data.append([float(e) for e in line[1:]])

        data = np.array(data)
        labels = np.array(labels).reshape(-1, 1)
        labels -= 1

        random_idxs = np.arange(len(data))
        np.random.shuffle(random_idxs)

        data = data[random_idxs]
        labels = labels[random_idxs]

        data = data[:5000]
        labels = labels[:5000]

        num_classes = 3

    # print(data)
    # print(labels)
    # print(data.shape)
    # print(labels.shape)
    # print(np.unique(labels, return_counts=True))
    # exit()

    return data, labels, num_classes
