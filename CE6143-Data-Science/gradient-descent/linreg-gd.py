#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Data Set: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
# Refs:
# http://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('gradient-descent/ENB2012_data.csv')
    feature_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    label_col = ['Y1']
    X = data[feature_col].as_matrix()
    y = data[label_col].as_matrix()

    # ref: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def gradient_descent(X, y, alpha = .001, iters = 100000, eps=1e-4):
    n, d = X.shape # (384, 8), n是資料總row數, d是feature數量col
    theta = numpy.matrix(numpy.zeros((d, 1))) # theta[8][1]: 8個feature的theta

    # DONE
    X_transpose = X.transpose() # 轉置矩陣
    for i in range(0, iters):
        last_theta = theta

        hypothesis = numpy.dot(X, theta)
        loss = hypothesis - y

        J = numpy.sum(numpy.square(loss)) / (2 * n)  # cost
        # print("Iteration %d | Cost: %.3f" % (i, J))

        gradient = numpy.dot(X_transpose, loss) / n
        theta = theta - alpha * gradient  # update

        # Use L1 norm to measure distance
        sum = 0
        for j in range(0, d):
            sum += (theta[j]-last_theta[j])
        if (sum < eps):
            # print(sum)
            # print("Iteration %d times" % i)
            break

    return theta


def predict(X, theta):
    return numpy.dot(X, theta)


def main(argv):
    # CVS有769筆資料 讀取並分割成X, y的train及test資料
    # X_train[384][8], X_test[384][8], y_train[384][1], y_test[384][1]
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)

    # 正規化 X的train跟test, 讓他介於0~1之間
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    # 使用Gradient descent解出theta是多少
    theta = gradient_descent(X_train_scale, y_train)

    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))

if __name__ == "__main__":
    main(sys.argv)
