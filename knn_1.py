# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import numpy as np
from sklearn.model_selection import train_test_split

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'new_dataset') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

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

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #print(classVotes)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    data = np.loadtxt("dataset.csv", delimiter=',')
    X = data[:,:]
    X_train, X_test = train_test_split(X, test_size = 0.0035)
# =============================================================================
# 	loadDataset('iris.data', split, trainingSet, testSet)
# 	print 'Train set: ' + repr(len(trainingSet))
# 	print 'Test set: ' + repr(len(testSet))
# =============================================================================
	# generate predictions
    predictions=[]
    k = 3
    for x in range(len(X_test)):
        neighbors = getNeighbors(X_train, X_test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('Test No. '+str(x+1)+'  > predicted=' + repr(result) + ', actual=' + repr(X_test[x][-1]))
    accuracy = getAccuracy(X_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
main()


