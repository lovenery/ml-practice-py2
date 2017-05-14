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
    data = pandas.read_csv('./HTRU2/HTRU_2.csv', header=None, names=['x%i' % (i) for i in xrange(8)] + ['y'])
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


def logreg_sgd(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # NOTE: compute theta
    n, d = X.shape  # (8949, 8)
    theta = numpy.zeros((d, 1)) # (8, 1)
    prev_loss = 2147483647
    for i in range(iters):
        yHat = predict_prob(X, theta) # (8949, 1)
        
        error = y.reshape(8949, 1) - yHat  # (8949,) - (8949, 1)
        theta = theta + alpha * X.T.dot(error) # (8, 8949) * (8949, 1)

        convert = lambda n : 1e-10 if n < 0.5 else (1 - 1e-10)
        prediction = map(convert, yHat)
        
        loss = cross_entropy(y, prediction)
        if ((prev_loss - loss) < eps):
            # print('iter: %d break' % i)
            break
        prev_loss = loss
    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # NOTE: compute tpr and fpr of different thresholds
    fpr = []
    tpr = []
    thresholds = numpy.linspace(min(y_prob), max(y_prob), 300)
    lysy = len(y_test) - sum(y_test)
    ys = sum(y_test)
    for i in thresholds:
        TP = 0.0
        FP = 0.0
        for j in range(len(y_prob)):
            if (y_prob[j] > i) and (y_test[j] == 0):
                FP += 1
            if (y_prob[j] > i) and (y_test[j] == 1):
                TP += 1
        fpr = numpy.insert(fpr, 0, FP / lysy)
        tpr = numpy.insert(tpr, 0, TP / ys)

    plt.title('ROC curve')
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig("./roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    # print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
