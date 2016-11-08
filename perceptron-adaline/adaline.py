
from __future__ import print_function

import numpy as np
import sys

from baselinearclassifier import BaseLinearClassifier


class AdalineGD(BaseLinearClassifier):

    """
    Adaline (ADAptive LInear NEuron) Classifier

    Uses Batch Gradient Decent (all of X to update weights)

    learningRate: between 0 and 1
        1 means the complete difference between actual and estimated is applied (might overestimate and/or never converge)
        0 means there is no learning (weights do not change at all)

    nIterations: integer >= 0
        Number of times (or "epochs") the weights are corrected
    """

    def __init__(self, learningRate=.01, nIterations=50):
        super(AdalineGD, self).__init__(learningRate, nIterations)
        self._cost = None

    @property
    def cost(self):
        if self._cost is None:
            print('Please fit() your model to generate the cost array', file=sys.stderr)
        return self._cost

    def fit(self, X, y):
        X, y = self._numpify_and_adjust(X, y)

        # initializing weights as an array of 0's for each feature (column), +1 for the threshold (w_0 where x_0 == 1)
        self._weights = np.zeros(1 + X.shape[1])
        self._cost = []

        for _ in xrange(self.nIterations):
            # 'output' is the dot product of the feature matrix and weights vector
            # it contains continuous predictions for all y values, before being binarized
            output = self._net_input(X)
            errors = y - output
            # Each column in X multiplied by the errors vector, summed, and then multiplied by the learning rate
            self._weights[1:] += self.learningRate * X.T.dot(errors)  # matrix-vector multiplication, one number per weight
            self._weights[0] += self.learningRate * errors.sum()
            cost = (1./2.) * (errors ** 2).sum()  # SSE (sum of squared errors)
            self._cost.append(cost)

        return self

    def _activation(self, X):
        """The Identity Matrix to _net_input()"""
        return self._net_input(X)

    def predict(self, X):
        return np.where(self._activation(X) >= 0, 1, -1)


class AdalineSGD(AdalineGD):

    """
    shuffle: bool (default=True)
        if True, training data will be shuffled after every epoch to prevent cycles (non-convergence)

    randomState: int or None (default=None)
        Use a state (seed) to repeat randomization of the training dataset
    """

    def __init__(self, learningRate=.01, nIterations=10, doShuffle=True, randomState=None):
        super(AdalineSGD, self).__init__(learningRate, nIterations)
        self.doShuffle = doShuffle
        if self.randomState is not None:
            np.random.seed(randomState)

    def fit(self, X, y):
        X, y = self._numpify_and_adjust(X, y)

        self._initialize_weights(X.shape[1])
        self._cost = []

        for _ in xrange(self.nIterations):
            if self.doShuffle:
                X, y = self._shuffle_training_data(X, y)

            cost = []
            for features, target in zip(X, y):
                cost.append(self._update_weights(features, target))

            self._cost.append(float(sum(cost)) / len(y))  # appending the average cost across all rows

        return self

    def partial_fit(self, X, y):
        """
        For Online learning; only reinitializes weights if they do not exist
        """
        X, y = self._numpify_and_adjust(X, y)

        if self._weights is None:
            self._initialize_weights(X.shape[1])

        for features, target in zip(X, y):
            self._update_weights(features, target)

        return self

    def _initialize_weights(self, m):
        self._weights = np.zeros(1 + m)

    @staticmethod
    def _shuffle_training_data(X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, features, target):
        # 'output' is the dot product of the feature vector and weights vector
        # it contains a continuous prediction for this row's y value, before being binarized
        output = self._net_input(features)
        error = target - output
        # same 'update' formula and usage as the Perceptron ('error' is no an integer here though)
        update = self.learningRate * error
        self._weights[1:] += update * features
        self._weights[0] += update
        cost = (1./2.) * (error ** 2)

        return cost
