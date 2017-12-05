import numpy as np
import pandas as pd

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)             #convert to NumPy matrix
    labelMat = np.mat(classLabels).T             #convert to NumPy matrix
    m, n = dataMatrix.shape

    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))

    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(np.dot(dataMatrix, weights))     #matrix mult
        error = labelMat - h                #vector subtraction
        weights = weights + alpha * np.dot(dataMatrix.T, error) #matrix mult
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = np.ones(n)

    for i in range(m):
        h = sigmoid(np.dot(dataMatrix[i], weights.T))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    import random

    m, n = dataMatrix.shape
    weights = np.ones(n)

    for j in range(numIter):
        dataIndex_list = random.sample(range(m), m)

        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001    #apha decreases with iteration, does not 
            randIndex = dataIndex_list[i]
            h = sigmoid(np.dot(dataMatrix[randIndex], weights.T))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]

    return weights


def plotBestFit(weights=None):
    import matplotlib.pyplot as plt

    dataMat, labelMat = loadDataSet()
    data_array = np.array(dataMat)
    n = data_array.shape[0]

    data_frame = pd.DataFrame(data_array)
    data_frame['label'] = pd.Series(labelMat, index=data_frame.index)

    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    xcord1.extend(data_frame[data_frame['label'] == 1][1])
    ycord1.extend(data_frame[data_frame['label'] == 1][2])

    xcord2.extend(data_frame[data_frame['label'] == 0][1])
    ycord2.extend(data_frame[data_frame['label'] == 0][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(np.dot(inX, weights.T))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest():
    fr_train_frame = pd.read_csv('horseColicTraining.txt', sep='\t', header=None, prefix='X')
    fr_test_frame = pd.read_csv('horseColicTest.txt', sep='\t', header=None, prefix='X')

    trainWeights = stocGradAscent1(fr_train_frame.iloc[:, 0:-1].values, fr_train_frame.iloc[:, -1], 1000)

    error_count = 0
    test_item_count = fr_test_frame.shape[0]
    for test_item in fr_test_frame.values:
        classify_result = classifyVector(np.array(test_item[0:-1]), trainWeights)
        if classify_result != test_item[-1]:
            error_count += 1

    errorRate = (float(error_count) / test_item_count)
    print('this time error rate = {0}'.format(errorRate))
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0

    for k in range(numTests):
        errorSum += colicTest()

    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':    
    dataMat, labelMat = loadDataSet()

    # weights = gradAscent(dataMat, labelMat)

    dataMatrix = np.array(dataMat)
    classLabels = np.mat(labelMat).T 
    # weights = stocGradAscent0(dataMatrix, classLabels)
    # weights = stocGradAscent1(dataMatrix, classLabels)

    multiTest()