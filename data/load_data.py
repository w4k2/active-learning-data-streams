import csv

import numpy as np
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


def get_data(dataset_name, seed_size, random_seed):
    X_train, X_test, y_train, y_test, num_classes, preprocessor = load_dataset(dataset_name, random_seed)
    X_stream, X_seed, y_stream, y_seed = sklearn.model_selection.train_test_split(X_train, y_train, test_size=seed_size, random_state=random_seed)

    X_seed = preprocessor.fit_transform(X_seed)
    X_stream = preprocessor.transform(X_stream)
    X_test = preprocessor.transform(X_test)

    stream = list(zip(X_stream, y_stream))
    return X_seed, y_seed, X_test, y_test, stream, num_classes


def load_dataset(dataset_name, random_seed):
    if dataset_name == 'accelerometer':
        return load_accelerometer(random_seed)
    elif dataset_name == 'adult':
        return load_adult()


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
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=random_seed)

    return X_train, X_test, y_train, y_test, num_classes, scaler


def load_adult():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'earnings']
    train_dataframe = pandas.read_csv('data/adult/adult.data', header=None, names=column_names)
    test_dataframe = pandas.read_csv('data/adult/adult.test', header=None, names=column_names, skiprows=1)

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    numeric_transformer = sklearn.pipeline.Pipeline(
        steps=[("imputer", sklearn.impute.SimpleImputer(strategy="median")), ("scaler", sklearn.preprocessing.StandardScaler())]
    )

    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    categorical_transformer = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")

    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0
    )

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
