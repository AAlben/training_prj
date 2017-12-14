import pandas as pd
import numpy as np
from numpy import *
import os
from scipy.spatial import distance


class KMeansExercise(object):       

    def __init__(self):
        pass 

    def loadDataSet(self, filename):
        data_frame = pd.read_csv(filename, sep='\t', header=None, prefix='X')
        return data_frame


    def distEclud(self, vecA, vecB):
        # return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
        return distance.cdist(vecA, vecB, 'minkowski', p=2.)


    def randCent(self, dataSet, k):
        m, n = dataSet.shape

        centroids = np.mat(np.zeros((k, n)))

        for j in range(n):
            minJ = np.min(dataSet[:, j])
            rangeJ = float(np.max(dataSet[:, j]) - minJ)
            centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))

        return centroids



    def kMeans(self, dataSet, k, distMeas=distEclud, createCent=randCent):
        m, n = dataSet.shape

        clusterAssment = np.mat(np.zeros((m, 2)))
        centroids = createCent(dataSet, k)

        clusterChanged = True

        while clusterChanged:
            clusterChanged = False

            for i in range(m):
                minDist = np.inf
                minIndex = -1

                for j in range(k):
                    distJI = distMeas(centroids[j, :], np.mat(dataSet[i, :]))

                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j

                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True

                clusterAssment[i, :] = np.array([minIndex, minDist ** 2])

            # print(centroids)

            for cent in range(k):
                ptsInclust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
                centroids[cent, :] = ptsInclust.mean(axis=0)

        return centroids, clusterAssment


def read_csv_file():
    header = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = '/home/lichenguang/code/Recommendation_Data/ml-20m'
    file_name = 'ratings.csv'
    data_frame = pd.read_csv(os.path.join(file_path, file_name), sep=',', names=header, skiprows=1)

    data_frame = data_frame.drop(['timestamp'], axis=1)
    return data_frame


def calculate_user_similar(data_frame, user_1_id, user_2_id):
    user_1_frame = data_frame[data_frame['userId'] == user_1_id]
    user_2_frame = data_frame[data_frame['userId'] == user_2_id]

    user_1_rated_movie = user_1_frame['movieId'].unique()
    user_2_rated_movie = user_2_frame['movieId'].unique()
    print('userid {0} user rated movie count = {1}'.format(user_1_id, user_1_rated_movie.shape[0]))
    print('userid {0} user rated movie count = {1}'.format(user_2_id, user_2_rated_movie.shape[0]))

    intersection_result = np.intersect1d(user_1_rated_movie, user_2_rated_movie)
    union_result = np.union1d(user_1_rated_movie, user_2_rated_movie)
    print('two user intersection count = {0}'.format(intersection_result.shape[0]))

    similar_result_version1 = len(intersection_result) * 1.0/ len(union_result)
    print(similar_result_version1)

    similar_result_version2 = len(intersection_result) * 1.0/ np.sqrt(len(user_1_rated_movie) * len(user_2_rated_movie))
    print(similar_result_version2)


def calculate_inverted_list(data_frame):
    data_frame = data_frame.iloc[0:1000]

    groupby_movie = data_frame.groupby('movieId')
    groupby_user = data_frame.groupby('userId')

    data_frame_pivot = data_frame.pivot(index='movieId', columns='userId', values='rating').dropna(axis=0, how='all')

    movie_item_with_user_matrix = data_frame_pivot.as_matrix()
    movie_item_with_user_matrix[np.isnan(movie_item_with_user_matrix)] = 0

    # use k-Means

    k_mean_exercise = KMeansExercise()
    centroids, clusterAssment = k_mean_exercise.kMeans(movie_item_with_user_matrix, 4, k_mean_exercise.distEclud, k_mean_exercise.randCent)

    print(clusterAssment)

    return None


    userid_list = data_frame['userId'].unique()[:10]
    # userid_list = [127607 121661  79011  26138 134047  17851  78914  67589  46580 120726]

    uniq_movie_id = np.array([], dtype=int)
    column_list = []
    for userid in userid_list:
        column_list.append('user_{0}'.format(userid))

    rated_movie_count = pd.Series(None, index=column_list)

    for userid in userid_list:
        user_rated_movie = data_frame[data_frame['userId'] == userid]['movieId'].unique()
        uniq_movie_id = np.union1d(uniq_movie_id, user_rated_movie)
        rated_movie_count['user_{0}'.format(userid)] = user_rated_movie.shape[0]

    '''
    rated_movie_count =
        user_1    175.0
        user_2    227.0
        user_3    364.0
        user_4    386.0
        dtype: float64
    '''

    inverted_frame = pd.DataFrame(None, index=uniq_movie_id, columns=column_list)

    for index in inverted_frame.index:
        rated_same_movie_users = data_frame[data_frame['movieId'] == index]['userId'].unique()

        for column in inverted_frame.keys():
            user_id = int(column.split('_')[-1])
            if user_id in rated_same_movie_users:
                inverted_frame.at[index, column] = 1

    '''
    inverted_frame =
              user_1 user_2 user_3 user_4
        1        NaN    NaN      1    NaN
        2          1    NaN    NaN    NaN
        3        NaN      1    NaN    NaN
        6        NaN    NaN    NaN      1
        10       NaN    NaN    NaN      1
        19       NaN    NaN    NaN      1
        24       NaN    NaN      1    NaN
        29         1    NaN    NaN    NaN
        32         1    NaN      1      1
    '''

    user_user_matricx = pd.DataFrame(None, index=column_list, columns=column_list)

    for userid in userid_list:
        one_user_frame = inverted_frame[inverted_frame['user_{0}'.format(userid)] == 1].sum()
        user_user_matricx.loc['user_{0}'.format(userid)] = one_user_frame
        user_user_matricx.loc['user_{0}'.format(userid)]['user_{0}'.format(userid)] = 0

    '''
    user_user_matricx = 
               user_1 user_2 user_3 user_4
        user_1      0      9     42      3
        user_2      9      0     15      2
        user_3     42     15      0      5
        user_4      3      2      5      0
    '''

    rated_movie_count = np.sqrt(rated_movie_count)

    division_result = user_user_matricx / rated_movie_count
    division_result = division_result.T / rated_movie_count
    division_result = division_result.T
    
    print(division_result)

    '''
    division_result = 
                   user_1     user_2     user_3     user_4
        user_1          0  0.0871081   0.232172  0.0428571
        user_2  0.0871081          0   0.140445  0.0483934
        user_3   0.232172   0.140445          0  0.0690987
        user_4  0.0428571  0.0483934  0.0690987          0
    '''
    return division_result, inverted_frame, userid_list


def recommend(user_similar_matricx, inverted_frame, recommend_item_K, recommend_user_id):
    user_id = recommend_user_id
    column_name = 'user_{0}'.format(user_id)

    user_similar = user_similar_matricx.loc['user_{0}'.format(user_id)]
    sorted_user_similar = user_similar.sort_values(ascending=False)
    K = 10
    user_similar_k = sorted_user_similar.index[:K]

    recommend_result = {}

    for recommend_item in inverted_frame.index:

        similar_sum = 0

        if inverted_frame.loc[recommend_item][column_name] == 1:
            continue

        for similar_user_id in user_similar_k:
            if inverted_frame.loc[recommend_item][similar_user_id] == 1:
                similar_sum += user_similar_matricx.loc[column_name][similar_user_id]

        recommend_result[recommend_item] = similar_sum

    recommend_result = pd.Series(recommend_result)
    sorted_recommend = recommend_result.sort_values(ascending=False)[:recommend_item_K]
    print(sorted_recommend)
    return sorted_recommend


def train_test_split_data(data_frame):
    from sklearn import model_selection
    train_data_frame, test_data_frame =  model_selection.train_test_split(data_frame, test_size=0.25)
    return train_data_frame, test_data_frame


if __name__ == '__main__':
    data_frame = read_csv_file()

    calculate_user_similar_flag = False

    if calculate_user_similar_flag:
        user_1_id = 1
        uniq_users = [3]

        for user_id in uniq_users:
            if user_id == 1:
                continue
            user_2_id = user_id
            print('*' * 10)
            print('userid = {0}'.format(user_id))
            print('*' * 10)
            calculate_user_similar(data_frame, user_1_id, user_2_id)

    train_data_frame, test_data_frame = train_test_split_data(data_frame)

    user_similar_matricx, inverted_frame, userid_list = calculate_inverted_list(train_data_frame)

    '''

    test_user_id = userid_list[0]
    test_data_user_like_movies = test_data_frame[test_data_frame['userId'] == test_user_id]['movieId'].unique()
    
    sorted_recommend = recommend(user_similar_matricx, inverted_frame, test_data_user_like_movies.shape[0], test_user_id)
    
    for movieId in sorted_recommend.index:
        if movieId in test_data_user_like_movies:
            print('this movieId = {0} is real like movie'.format(movieId))

    '''
