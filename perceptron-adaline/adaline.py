
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
        X = np.array(X)
        y = np.array(y)

        # adjust target variable to fit perceptron (if needed)
        y = np.where(y == 1, 1, -1)

        # initializing weights as an array of 0's for each feature (column), +1 for the threshold (w_0 where x_0 == 1)
        self._weights = np.zeros(1 + X.shape[1])
        self._cost = []

        for _ in xrange(self.nIterations):
            yPred = self._net_input(X)  # this is the predicted y's (y-hats) for every row
            errors = y - yPred
            # Each column in X multiplied by the errors vector, summed, and then multiplied by the learning rate
            self._weights[1:] += self.learningRate * X.T.dot(errors)  # matrix-vector multiplication, one number per weight
            self._weights[0] += self.learningRate * errors.sum()
            cost = (1./2.) * (errors ** 2).sum()  # SSE (sum of squared errors)
            self._cost.append(cost)

        return self

    def _activation(self, X):
        return self._net_input(X)

    def predict(self, X):
        return np.where(self._activation(X) >= 0, 1, -1)
