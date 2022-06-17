import numpy as np
from sklearn.base import clone
from strlearn.ensembles.base import StreamingEnsemble


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
