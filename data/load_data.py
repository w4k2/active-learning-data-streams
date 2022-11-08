import csv
from random import random

import numpy as np
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


def get_data(dataset_name, random_seed):
    X_train, X_test, y_train, y_test, num_classes, preprocessor = load_dataset(dataset_name, random_seed)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, y_train, X_test, y_test, num_classes


def load_dataset(dataset_name, random_seed):
    if dataset_name == 'accelerometer':
        return load_accelerometer(random_seed)
    elif dataset_name == 'adult':
        return load_adult()
    elif dataset_name == 'bank_marketing':
        return load_bank(random_seed)
    elif dataset_name == 'firewall':
        return load_firewall(random_seed)
    elif dataset_name == 'chess':
        return load_chess(random_seed)
    elif dataset_name == 'nursery':
        return load_nursery(random_seed)
    elif dataset_name == 'poker':
        return load_poker()
    elif dataset_name == 'mushroom':
        return load_mushroom(random_seed)
    elif dataset_name == 'wine':
        return load_wine(random_seed)
    elif dataset_name == 'abalone':
        return load_abalone(random_seed)
    else:
        raise ValueError("Invalid dataset name")


def load_accelerometer(random_seed):
    X = []
    y = []
    with open('data/accelerometer/accelerometer.csv', 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            y.append(float(line[0]))
            X.append([float(e) for e in line[1:]])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    y -= 1

    random_idxs = np.arange(len(X))
    np.random.shuffle(random_idxs)

    X = X[random_idxs]
    y = y[random_idxs]

    X = X[:5000]
    y = y[:5000]

    num_classes = 3
    scaler = sklearn.preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    return X_train, X_test, y_train, y_test, num_classes, scaler


def load_adult():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'earnings']
    train_dataframe = pandas.read_csv('data/adult/adult.data', header=None, names=column_names)
    test_dataframe = pandas.read_csv('data/adult/adult.test', header=None, names=column_names, skiprows=1)

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X_train = train_dataframe.loc[:, train_dataframe.columns != 'earnings']
    y_train = train_dataframe.loc[:, train_dataframe.columns == 'earnings']
    y_train = y_train.replace([' <=50K', ' >50K'], [0, 1])
    y_train = y_train.to_numpy().reshape(-1, 1)
    X_test = test_dataframe.loc[:, test_dataframe.columns != 'earnings']
    y_test = test_dataframe.loc[:, test_dataframe.columns == 'earnings']
    y_test = y_test.replace([' <=50K.', ' >50K.'], [0, 1])
    y_test = y_test.to_numpy().reshape(-1, 1)

    num_classes = 2

    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_bank(random_seed):
    df = pandas.read_csv('data/bank_marketing/bank-full.csv', delimiter=';')

    numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', ]
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'y']
    y = df.loc[:, 'y']
    y = y.replace(['no', 'yes'], [0, 1])
    y = y.to_numpy().reshape(-1, 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 2

    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_firewall(random_seed):
    df = pandas.read_csv('data/firewall/log2.csv')
    df = df[df.loc[:, 'Action'] != 'reset-both']

    numeric_features = ['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port',
                        'Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']
    categorical_features = []
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'Action']
    y = df.loc[:, 'Action']
    y = y.replace(['allow', 'deny', 'drop', 'reset-both'], [0, 1, 2, 3])
    y = y.to_numpy().reshape(-1, 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 3

    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_chess(random_seed):
    df = pandas.read_csv('data/chess/krkopt.data', header=None)
    df = df[df.iloc[:, 6] != 'zero']
    df = df[df.iloc[:, 6] != 'one']
    df = df[df.iloc[:, 6] != 'three']

    numeric_features = []
    categorical_features = [0, 1, 2, 3, 4, 5]
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 6]
    y = df.loc[:, 6]

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 17

    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_nursery(random_seed):
    column_names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'CLASS']
    df = pandas.read_csv('data/nursery/nursery.data', header=None, names=column_names)
    df = df[df.loc[:, 'CLASS'] != 'recommend']

    numeric_features = []
    categorical_features = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social']
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'CLASS']
    # drop most informative columns
    X = df.loc[:, df.columns != 'health']
    y = df.loc[:, 'CLASS']

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 4
    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_poker():
    column_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
    train_df = pandas.read_csv('data/poker/poker-hand-testing.data', header=None, names=column_names)  # for some reason test .data file has much more samples the train
    train_df = train_df.iloc[:20000, :]
    test_df = pandas.read_csv('data/poker/poker-hand-training-true.data', header=None, names=column_names)
    train_df = train_df[train_df.loc[:, 'CLASS'] != 7]
    train_df = train_df[train_df.loc[:, 'CLASS'] != 8]
    train_df = train_df[train_df.loc[:, 'CLASS'] != 9]
    test_df = test_df[test_df.loc[:, 'CLASS'] != 7]
    test_df = test_df[test_df.loc[:, 'CLASS'] != 8]
    test_df = test_df[test_df.loc[:, 'CLASS'] != 9]

    numeric_features = ['C1', 'C2', 'C3', 'C4', 'C5']
    categorical_features = ['S1', 'S2', 'S3', 'S4', 'S5']
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X_train = train_df.loc[:, train_df.columns != 'CLASS']
    y_train = train_df.loc[:, 'CLASS']
    X_test = test_df.loc[:, test_df.columns != 'CLASS']
    y_test = test_df.loc[:, 'CLASS']

    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)
    class_econder = sklearn.preprocessing.OrdinalEncoder()
    y_train = class_econder.fit_transform(y_train)
    y_test = class_econder.transform(y_test)

    num_classes = 10
    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_mushroom(random_seed):
    column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                    'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    df = pandas.read_csv('data/mushroom/agaricus-lepiota.data', header=None, names=column_names)

    numeric_features = []
    categorical_features = ['cap-shape', 'cap-color']
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'class']
    # drop most informative columns
    X = df.loc[:, df.columns != 'bruises']
    X = df.loc[:, df.columns != 'cap-surface']
    X = df.loc[:, df.columns != 'gill-spacing']
    X = df.loc[:, df.columns != 'gill-size']
    X = df.loc[:, df.columns != 'gill-color']
    X = df.loc[:, df.columns != 'stalk-root']
    X = df.loc[:, df.columns != 'stalk-surface-above-ring']
    X = df.loc[:, df.columns != 'stalk-surface-below-ring']
    X = df.loc[:, df.columns != 'stalk-color-above-ring']
    X = df.loc[:, df.columns != 'veil-type']
    X = df.loc[:, df.columns != 'ring-type']
    X = df.loc[:, df.columns != 'ring-number']
    X = df.loc[:, df.columns != 'spore-print-color']
    X = df.loc[:, df.columns != 'population']
    X = df.loc[:, df.columns != 'habitat']
    X = df.loc[:, df.columns != 'veil-color']
    X = df.loc[:, df.columns != 'stalk-color-below-ring']
    X = df.loc[:, df.columns != 'gill-attachment']
    X = df.loc[:, df.columns != 'stalk-shape']
    X = df.loc[:, df.columns != 'odor']

    y = df.loc[:, 'class']

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 2
    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_wine(random_seed):
    df = pandas.read_csv('data/wine/winequality-white.csv', delimiter=';')
    df = df[df.loc[:, 'quality'] != 3]
    df = df[df.loc[:, 'quality'] != 9]

    numeric_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    categorical_features = []
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'quality']
    y = df.loc[:, 'quality']

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 5
    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def load_abalone(random_seed):
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    df = pandas.read_csv('data/abalone/abalone.data', header=None, names=column_names)
    for c in [32, 20, 3, 21, 23, 22, 27, 24, 1, 26, 29, 2, 25]:
        df = df[df.loc[:, 'Rings'] != c]

    numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
    categorical_features = ['Sex']
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != 'Rings']
    y = df.loc[:, 'Rings']

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    num_classes = 16
    return X_train, X_test, y_train, y_test, num_classes, preprocessor

    # return load_pandas(
    #     'data/abalone/abalone.data',
    #     random_seed,
    #     ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'],
    #     ['Sex'],
    #     'Rings',
    #     column_names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    # )


def load_pandas(filepath, random_seed, numeric_features, categorical_features, target_feature, column_names=None, delimiter=None):
    header = 'infer'
    if column_names is not None:
        header = None

    df = pandas.read_csv(filepath, header=header, names=column_names, delimiter=delimiter)
    counts_df = pandas.value_counts(df.loc[:, target_feature])
    low_count_classes = list(counts_df.loc[counts_df < 50].index)
    for class_to_remove in low_count_classes:
        df = df[df.loc[:, target_feature] != class_to_remove]

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    X = df.loc[:, df.columns != target_feature]
    y = df.loc[:, target_feature]

    y = y.to_numpy().reshape(-1, 1)
    y = sklearn.preprocessing.OrdinalEncoder().fit_transform(y)
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed, stratify=y)
    return X_train, X_test, y_train, y_test, num_classes, preprocessor


def get_preprocessor(numeric_features, categorical_features):
    numeric_transformer = sklearn.pipeline.Pipeline(
        steps=[("imputer", sklearn.impute.SimpleImputer(strategy="median")), ("scaler", sklearn.preprocessing.StandardScaler())]
    )

    categorical_transformer = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")

    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0
    )

    return preprocessor


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, num_classes, preprocessor = load_nursery(42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(X_train.shape)

    df = pandas.DataFrame(np.concatenate((X_train, y_train), axis=1))
    correlation = df.corr().iloc[:, -1].abs()
    print(correlation)
    print(correlation.sort_values())

    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import balanced_accuracy_score
    model = MLPClassifier(hidden_layer_sizes=(100, 100))
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, test_pred)
    print('acc = ', acc)
