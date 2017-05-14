#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey (South) [1]. 
1. Mean of the integrated profile. 
2. Standard deviation of the integrated profile. 
3. Excess kurtosis of the integrated profile. 
4. Skewness of the integrated profile. 
5. Mean of the DM-SNR curve. 
6. Standard deviation of the DM-SNR curve. 
7. Excess kurtosis of the DM-SNR curve. 
8. Skewness of the DM-SNR curve. 
9. Class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    # column name: x0, x1..., x7, y
    data = pandas.read_csv('./logreg-sgd/HTRU2/HTRU_2.csv', header=None, names=['x%i' % (i) for i in xrange(8)] + ['y'])
    # split data to X and y
    X = numpy.asarray(data[['x%i' % (i) for i in xrange(8)]])
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def cross_entropy(y, y_hat):
    loss = 0
    for i in xrange(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss


# logistic regression using stocastic gradient decent
def logreg_sgd(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # NOTE: compute theta
    # n, d = X.shape  # (8949, 8)
    # theta = numpy.zeros((d, 1)) # (8, 1)
    # using log loss (cross-entropy loss)
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log", alpha=alpha, epsilon=eps)
    clf.fit(X, y)
    theta = clf.coef_.transpose()
    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # NOTE: compute tpr and fpr of different thresholds
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.title('ROC curve')
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    # plt.savefig("./logreg-sgd/roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
