# http://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
# yhat = e^(b0 + b1 * x1) / ( 1 + e^(b0 + b1 * x1) )
# yhat = 1.0 / ( 1.0 + e^(-(b0 + b1 * x1)) )
# b = b + learning_rate * (Y - yhat) * yhat * (1 - yhat) * X
# Data Set:
# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
# [768][9]
import matplotlib.pyplot as plt
import pandas as pd
import math

# Make a prediction with coefficients
def predict(row, coefficients): # row=1d list, coefficients=1d list
    # b0 + b1*x1 + b2*x2
    yhat = coefficients[0] # b0
    for i in range(len(row) - 1): # run every column of data # 3-1=2, range 0~1
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + math.exp(-yhat))

dataset = [
    [2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]
]

# test predictions
# coef = [-0.406605464, 0.852573316, -1.104746259] # fake coef
# for row in dataset:
#     yhat = predict(row, coef)
#     print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))

# Plot fake dataset
# d = pd.DataFrame(dataset)
# X1 = d[0].as_matrix()
# X2 = d[1].as_matrix()
# class0 = plt.scatter(X1[:5], X2[:5])
# class1 = plt.scatter(X1[5:], X2[5:], color='red')
# plt.grid(axis='y')
# plt.legend((class0, class1), ('Class 0', 'Class 1'), scatterpoints=1, loc='lower left')
# plt.show()

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))] # init to 0
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat # real-predict=error
            sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat) # first coef
            for i in range(len(row)-1): # column loop
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef

# Calculate coefficients
# dataset = [[2.7810836,2.550537003,0],
#     [1.465489372,2.362125076,0],
#     [3.396561688,4.400293529,0],
#     [1.38807019,1.850220317,0],
#     [3.06407232,3.005305973,0],
#     [7.627531214,2.759262235,1],
#     [5.332441248,2.088626775,1],
#     [6.922596716,1.77106367,1],
#     [8.675418651,-0.242068655,1],
#     [7.673756466,3.508563011,1]]
# l_rate = 0.3
# n_epoch = 100
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)

###################################################
# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)

# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
import os.path as path
filename = path.join(path.dirname(__file__), './pima-indians-diabetes.csv')
dataset = load_csv(filename)
for i in range(len(dataset[0])): # range 0~8, 9
    str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))