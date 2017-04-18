import csv
import random
import math
import operator
import matplotlib.pyplot as plt

def loadDataset(filename, split, trainingSet = [] , testSet = []):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)  # 151 last line is blank
        for x in range(len(dataset)-1):  # 0~149
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
# trainingSet = []
# testSet = []
# loadDataset('iris.data', 0.66, trainingSet, testSet)
# print 'Train: ' + repr(len(trainingSet))
# print 'Test: ' + repr(len(testSet))


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
# data1 = [2, 2, 2, 'a']
# data2 = [4, 4, 4, 'b']
# distance = euclideanDistance(data1, data2, 3)
# print 'Distance: ' + repr(distance)

# kNN
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbors = getNeighbors(trainSet, testInstance, 1)
# print(neighbors)

# voting
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
# neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# response = getResponse(neighbors)
# print(response)

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)

trainingSetError = []
testSetError = []
trainingSetAccuracy = []
testSetAccuracy = []

# prepare data
trainingSet = []
testSet = []
split = 0.50
loadDataset('knn1/iris.data', split, trainingSet, testSet)
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))
def main(k):
    # generate predictions
    predictions = []
    trainPredictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print '> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])
    for x in range(len(trainingSet)):
        neighbors = getNeighbors(trainingSet, trainingSet[x], k)
        result = getResponse(neighbors)
        trainPredictions.append(result)
        # print '> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])

    accuracy = getAccuracy(testSet, predictions)
    trainAccuracy = getAccuracy(trainingSet, trainPredictions)

    testSetError.append(len(testSet) * (1 - accuracy / 100))
    trainingSetError.append(len(trainingSet) * (1 - trainAccuracy / 100))
    trainingSetAccuracy.append(trainAccuracy)
    testSetAccuracy.append(accuracy)

    print 'TrainingSetError: ' + repr(trainingSetError)
    print 'TestSetError: ' + repr(testSetError)
    print 'TrainAccuracy: ' + repr(trainAccuracy)
    print 'Accuracy: ' + repr(accuracy) + '%\n'


for x in range(1, 21):
    print 'k=%d' % x
    main(x)

plt.plot(['%d' % x for x in range(1, 21)], trainingSetError)
plt.plot(['%d' % x for x in range(1, 21)], testSetError)
plt.grid(True)
plt.ylabel('Error Number')
plt.title('Learning curve - k vs error')
plt.show()

plt.plot(['%d' % x for x in range(1, 21)], trainingSetAccuracy)
plt.plot(['%d' % x for x in range(1, 21)], testSetAccuracy)
plt.grid(True)
plt.ylabel('Performance(%)')
plt.title('Learning curve - k vs accuracy')
plt.show()
