
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, testIndexRange=None, resolution=0.02):
    """
    Visualize Decision Boundaries
    """
    # Marker Generator & Color Map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('orange', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the Surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # x-axis
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # y-axis

    # arange() creates a 1-D array from min to max with resolution increments
    # meshgrid() turns xarray into xmatrix with len(yarray) rows, each row = xarray
    #            turns yarray into ymatrix with len(yarray) rows, each row repeats yarray element len(xarray) times
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # ravel() flattens the array
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)  # reshapes without changing data

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot Class Samples
    for e, target in enumerate(np.unique(y)):
        plt.scatter(x=X[y == target, 0], y=X[y == target, 1], alpha=0.8,
                    c=cmap(e), marker=markers[e], label=target)

    if testIndexRange:
        XTest = X[testIndexRange, :]
        plt.scatter(XTest[:, 0], XTest[:, 1], c='', alpha=1.0, linewidths=1,
                    marker='o', s=55, label='test set')

    return plt
