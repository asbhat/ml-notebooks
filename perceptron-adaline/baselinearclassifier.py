
from __future__ import print_function

import numpy as np
import sys

from abc import ABCMeta, abstractmethod

class BaseLinearClassifier(object):

    """
    The base class for linear classifiers

    learning_rate: between 0 and 1
        1 means the complete difference between actual and estimated is applied (might overestimate and/or never converge)
        0 means there is no learning (weights do not change at all)

    iterations: integer >= 0
        Number of times (or "epochs") the weights are corrected
    """

    __metaclass__ = ABCMeta

    def __init__(self, learning_rate=.01, iterations=10):
        self.learning_rate = self._check_learning_rate(learning_rate)
        self.iterations = self._check_n_iter(iterations)
        self._weights = None
        self._errors = None

    @staticmethod
    def _check_learning_rate(learning_rate):
        if learning_rate < 0 or learning_rate > 1:
            raise ValueError('learning_rate must be between 0.0 and 1.0')
        return learning_rate

    @staticmethod
    def _check_n_iter(iterations):
        try:
            iterations = int(iterations)
        except ValueError as e:
            raise 'iterations must be an integer\n\n{e}'.format(e=e)
        if iterations < 0:
            raise ValueError('iterations must be greater than or equal to zero')
        return iterations

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

    @abstractmethod
    def fit(self, X, y):
        pass

    @staticmethod
    def _numpify_and_adjust(X, y):
        """
        Adjusts the X matrix and y vector before fitting

        1) Makes sure both X and y are np arrays
        2) Adjusts the y target variable (can often be 1's and 0's) to 1 and -1
        """
        X = np.array(X)
        y = np.array(y)
        # adjust target variable to fit perceptron (if needed)
        y = np.where(y == 1, 1, -1)
        return X, y

    def predict(self, X):
        """Predict class label based on current weights"""
        return np.where(self._net_input(X) >= 0, 1, -1)

    def _net_input(self, X):
        """Dot product of current weights and features (X)"""
        return np.dot(X, self._weights[1:]) + self._weights[0]  # w_0 is the threshold
