
from __future__ import print_function

import numpy as np
import sys


class Perceptron(object):

    """
    A Perceptron (linear) classifier

    learningRate: between 0 and 1
        1 means the complete difference between actual and estimated is applied (might overestimate and/or never converge)
        0 means there is no learning (weights do not change at all)

    nIterations: integer >= 0
        Number of times (or "epochs") the weights are corrected
    """

    def __init__(self, learningRate=.01, nIterations=10):
        self.learningRate = self._check_learning_rate(learningRate)
        self.nIterations = self._check_n_iter(nIterations)
        self._weights = None
        self._errors = None

    @staticmethod
    def _check_learning_rate(learningRate):
        if learningRate < 0 or learningRate > 1:
            raise ValueError('learningRate must be between 0.0 and 1.0')
        return learningRate

    @staticmethod
    def _check_n_iter(nIterations):
        try:
            nIterations = int(nIterations)
        except ValueError as e:
            raise 'nIterations must be an integer\n\n{e}'.format(e=e)
        if nIterations < 0:
            raise ValueError('nIterations must be greater than or equal to zero')
        return nIterations

    @property
    def weights(self):
        if self._weights is None:
            print('Please fit() your model to generate weights', file=sys.stderr)
        return self._weights

    @property
    def errors(self):
        if self._errors is None:
            print('Please fit() your model to generate errors', file=sys.stderr)
        return self._errors

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # adjust target variable to fit perceptron (if needed)
        y = np.where(y == 1, 1, -1)

        # initializing weights as an array of 0's for each feature (column), +1 for the threshold (w_0 where x_0 == 1)
        self._weights = np.zeros(1 + X.shape[1])
        self._errors = []

        for _ in xrange(self.nIterations):
            errorCount = 0
            for features, target in zip(X, y):
                update = self.learningRate * (target - self.predict(features))
                self._weights[1:] += update * features  # update weights for this row's features
                self._weights[0] += update  # update the threshold
                errorCount += int(update != 0)  # count the rows in X where target != predicted

            self._errors.append(errorCount)  # one value per iteration

        return self

    def predict(self, X):
        """Predict class label based on current weights"""
        return np.where(self._net_input(X) >= 0, 1, -1)

    def _net_input(self, X):
        """Dot product of current weights and features (X)"""
        return np.dot(X, self._weights[1:]) + self._weights[0]  # w_0 is the threshold
