import numpy as np
import numpy as np
from sklearn.base import clone
from strlearn.ensembles.base import StreamingEnsemble
from strlearn.streams import StreamGenerator
from sklearn.model_selection import train_test_split


class OnlineBagging(StreamingEnsemble):
    """
    Online Bagging.
    """

    def __init__(self, base_estimator=None, n_estimators=10):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.was_trained = False

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        if len(self.ensemble_) == 0:
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        self.weights = []
        for instance in range(self.X_.shape[0]):
            K = np.asarray(
                [np.random.poisson(1, 1)[0] for i in range(self.n_estimators)]
            )
            self.weights.append(K)

        self.weights = np.asarray(self.weights).T

        for w, base_model in enumerate(self.ensemble_):
            classes = None if self.was_trained else self.classes_
            base_model.partial_fit(
                self.X_, self.y_, classes, sample_weight=self.weights[w]
            )

        self.was_trained = True

        return self


def get_data(stream_len, seed_size, random_seed):
    stream = StreamGenerator(
        n_chunks=stream_len,
        chunk_size=1,
        n_drifts=0,
        random_state=random_seed,
        y_flip=0.0,
        n_classes=3,
        n_features=5,
        n_informative=5,
        n_redundant=0,
    )

    seed_data, seed_target, stream = select_seed(stream, seed_size)
    data, target = zip(*stream)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42, stratify=target)
    train_stream = list(zip(X_train, y_train))
    X_test = np.squeeze(X_test)

    return seed_data, seed_target, train_stream, X_test, y_test


def select_seed(stream, seed_size):
    data = []
    target = []
    new_stream = []

    while len(data) < seed_size:
        X, y = stream.get_chunk()
        data.append(X)
        target.append(y)

    while not stream.is_dry():
        batch = stream.get_chunk()
        new_stream.append(batch)

    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)

    return data, target, new_stream
