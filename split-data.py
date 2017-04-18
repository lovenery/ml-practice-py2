import sklearn.model_selection

test = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [15, 16]]
train, test = sklearn.model_selection.train_test_split(test, train_size = 0.66)
print(train)
print(test)