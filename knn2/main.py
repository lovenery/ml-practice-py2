import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import scipy.spatial.distance
import operator
# import xlrd # pip install xlrd
# ref: 
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# http://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/


def getCosineDistance(a, b):
    """
    Return 
    sortedVotes[0][0]: str
    
    Parameters
    a, b : numpy array
        last item is a str
        np.array([1., 2., 2., 'a'])
    """
    a = a[:-1].astype(float)
    b = b[:-1].astype(float)
    distance = 1. - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    # distance = scipy.spatial.distance.cosine(a, b)
    return distance
# data1 = np.array([1., 2., 2., 'a'])
# data2 = np.array([2., 2, 1., 'b'])
# distance = getCosineDistance(data1, data2)
# print 'Distance: ' + repr(distance)


def knn(trainSet, test, k):
    """
    Return 
    distance: str
    
    Parameters
    trainSet : padnas.DataFrame
    test : pandas.Series
    k : int
    """
    ### Get neighbors
    distances = []
    for x in range(len(trainSet)):
        train = trainSet.iloc[x]
        dist = getCosineDistance(np.array(test), np.array(train))
        distances.append((train, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    # NOTE
    # type(neighbors) == list
    # type(neighbors[x]) == pandas.series

    ### Voting
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x].iloc[-1] # label
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(trainSet, testSet, k):
    """
    Return 
    accuracy: float
    
    Parameters
    trainSet : padnas.DataFrame
    testSet : padnas.DataFrame
    k : int
    """
    ### Get predictions
    predictions = []
    for x in range(len(testSet)):
        response = knn(trainSet, testSet.iloc[x], k)
        predictions.append(response)
        # print('predicted = ' + response + ', actual=' + testSet.iloc[x, -1])

    ### Evaulate accuracy
    correct = 0
    for x in range(len(testSet)):
        if testSet.iloc[x, -1] is predictions[x]:
            correct += 1
    accuracy = ( correct / float(len(testSet)) ) * 100.0
    return accuracy


def main():
    ### Loading data
    excel = pd.read_excel('knn2/data.xlsx', header = None)
    train, test = sklearn.model_selection.train_test_split(excel, train_size = 0.66)

    # you can test
    # print(excel)
    # print(train) # DataFrame
    # print('#################################')
    # print(test) # DataFrame
    # how to get row element
    # print(test.iloc[0]) 
    # print(test.iloc[0, -1]) 

    accuracies = []
    for k in range(1, 31):
        accuracy = getAccuracy(train, test, k)
        print('K = '  + repr(k).ljust(2) + ' Accuracy: ' + repr(accuracy) + ' %')
        accuracies.append(accuracy)

    # picture
    k_range = range(1, 31)
    plt.plot(k_range, accuracies)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

main()

# comment main() you can test
# d = {
#     'one': [2., 0., 3.],
#     'two': [2., 0., 3.],
#     'three': [1., 1., 3.],
#     'four': ['a', 'a', 'c']
# }
# dummy_train = pd.DataFrame(d, columns = ['one', 'two', 'three', 'four'])
# d = {
#     'one': [2.],
#     'two': [2.],
#     'three': [1.],
#     'four': ['d']
# }
# dummy_test = pd.DataFrame(d, columns = ['one', 'two', 'three', 'four'])
# res = knn(dummy_train, dummy_test.iloc[0], 3)
# print('This test data predict by knn is: ' + repr(res))
# print('But actually it is: ' + dummy_test.iloc[0, -1])
