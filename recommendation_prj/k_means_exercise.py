import pandas as pd
import numpy as np
import os


def kMeans(dataSet, k):
    m, n = dataSet.shape
    print('m = {0} ; n = {1}'.format(m, n))

    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = np.mat(np.zeros((k, n)))

    for index in range(k):
        centroids[:, index] = np.mat(5 + 5 * np.random.rand(k, 1))

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False

        for i in range(m):
            minDist = 0
            minIndex = -1

            for j in range(k):
                vecA = np.array(centroids)[j, :]
                vecB = np.array(dataSet)[i, :]
                distJI = np.sqrt(sum(np.power(vecA - vecB, 2)))

                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):
            print(dataSet[np.nonzero(np.array(clusterAssment)[:, 0] == cent)])
            break


            ptsInClust = dataSet[np.nonzero(np.array(clusterAssment)[:, 0] == cent)][0][0]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

    id = np.nonzero(np.array(clusterAssment)[:, 0] == cent)[0]
    return centroids, clusterAssment, id


def load_data_file(file_path, file_name):
    file_name = os.path.join(file_path, file_name)
    data_frame = pd.read_csv(file_name, sep=',', header=None)
    return data_frame


if __name__ == '__main__':
    file_path = '/home/lichenguang/code/KNN_Data'
    file_name_list = ['bezdekIris.data', 'iris.data']

    bezdekIris_frame = load_data_file(file_path, file_name_list[0])
    iris_frame = load_data_file(file_path, file_name_list[1])

    centroids, clusterAssment, id = kMeans(bezdekIris_frame.ix[:, 0:bezdekIris_frame.shape[1] - 2].as_matrix(), 2)

    print(centroids)