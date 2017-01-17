
import numpy as np

from baselinearclassifier import BaseLinearClassifier

class Perceptron(BaseLinearClassifier):

    """
    A Perceptron (linear) classifier

    learning_rate: between 0 and 1
        1 means the complete difference between actual and estimated is applied (might overestimate and/or never converge)
        0 means there is no learning (weights do not change at all)

    iterations: integer >= 0
        Number of times (or "epochs") the weights are corrected
    """

    def __init__(self, learning_rate=.01, iterations=10):
        super(Perceptron, self).__init__(learning_rate, iterations)

    def fit(self, X, y):
        X, y = self._numpify_and_adjust(X, y)

        # initializing weights as an array of 0's for each feature (column), +1 for the threshold (w_0 where x_0 == 1)
        self._weights = np.zeros(1 + X.shape[1])
        self._errors = []

        for _ in xrange(self.iterations):
            error_count = 0
            for features, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(features))
                self._weights[1:] += update * features  # update weights for this row's features
                self._weights[0] += update  # update the threshold
                error_count += int(update != 0)  # count the rows in X where target != predicted

            self._errors.append(error_count)  # one value per iteration

        return self
