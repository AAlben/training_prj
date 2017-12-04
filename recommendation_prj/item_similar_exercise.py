import pandas as pd
import numpy as np
import os

from itertools import combinations
import random
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist


def read_csv_file():
    header = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = '/home/lichenguang/code/Recommendation_Data/ml-20m'
    file_name = 'ratings_5000.csv'
    # file_name = 'ratings.csv'
    data_frame = pd.read_csv(os.path.join(file_path, file_name), sep=',', names=header, skiprows=1)

    data_frame = data_frame.drop(['rating', 'timestamp'], axis=1)
    return data_frame


def calculate_inverted_list(data_frame):
    groupby_movie = data_frame.groupby('movieId')
    groupby_user = data_frame.groupby('userId')

    data_frame['value'] = pd.Series(1, index=data_frame.index)
    data_frame_pivot = data_frame.pivot(index='userId', columns='movieId', values='value').dropna(axis=1, how='all')

    movie_index_series = pd.Series(range(len(data_frame_pivot.columns)), index=data_frame_pivot.columns)
    # print(movie_index_series)
    '''
        movieId
        1            0
        2            1
        3            2
        5            3
        6            4
        7            5
        8            6
        9            7
        10           8
        11           9    
    '''

    groupby_user_count = groupby_user.count().sort_values(by='movieId', ascending=False)
    # print(groupby_user_count[groupby_user_count['movieId'] > 100])

    '''
    max_count_movieId = 480
    max_count_userId = 11
    print(data_frame[data_frame['userId'] == max_count_userId])
          userId  movieId  value
    992       11      384      1
    1429      11    60514      1   
    ''' 

    movie_item_similar_matrix = data_frame_pivot.values
    '''
        movieId
        1         1.0
        39        1.0
    '''

    movie_item_similar_matrix[np.isnan(movie_item_similar_matrix)] = 0
    movie_similar_result_matrix = np.dot(movie_item_similar_matrix.T, movie_item_similar_matrix)
    '''
        [[ 11.   1.   2. ...,   1.   1.   1.]
         [  1.   3.   0. ...,   0.   0.   0.]
         [  2.   0.   7. ...,   0.   0.   0.]
         ..., 
         [  1.   0.   0. ...,   1.   1.   1.]
         [  1.   0.   0. ...,   1.   1.   1.]
         [  1.   0.   0. ...,   1.   1.   1.]]
        length = (1757, 1757)
    '''

    denominator = np.diag(movie_similar_result_matrix)
    matrix_diag = np.diag(-1 * denominator)
    movie_similar_diag_0_matrix = movie_similar_result_matrix + matrix_diag

    movie_similar_diag_0_matrix = 1.0 * movie_similar_diag_0_matrix / denominator
    movie_similar_diag_0_matrix = movie_similar_diag_0_matrix.T / denominator
    movie_similar_diag_0_matrix = movie_similar_diag_0_matrix.T
    movie_similar_result_frame = pd.DataFrame(movie_similar_diag_0_matrix, index=data_frame_pivot.columns, columns=data_frame_pivot.columns)

    print(movie_similar_diag_0_matrix)

    return movie_similar_result_frame, groupby_user


def recommend(movie_similar_result_frame, groupby_user, recommend_item_K, recommend_user_id):
    K = recommend_item_K

    recommend_result = {}

    user_rated_movieId_list = groupby_user.get_group(recommend_user_id)['movieId']
    for user_rated_movieId in user_rated_movieId_list:
        with_other_item_similar = movie_similar_result_frame[user_rated_movieId]
        sorted_similar = with_other_item_similar.sort_values(ascending=False).iloc[:K]

        for sorted_similar_movieId in sorted_similar.keys():
            if sorted_similar_movieId in user_rated_movieId_list:
                continue
            recommend_result[sorted_similar_movieId] = recommend_result.get(sorted_similar_movieId, 0)
            recommend_result[sorted_similar_movieId] += sorted_similar[sorted_similar_movieId]

    recommend_result_series = pd.Series(recommend_result)
    # print(recommend_result_series)
    return recommend_result_series


def train_test_split_data(data_frame):
    from sklearn import model_selection
    train_data_frame, test_data_frame =  model_selection.train_test_split(data_frame, test_size=0.25)
    return train_data_frame, test_data_frame


class CheckAlgorithm(object):

    def __init__(self):
        pass

    def check_precision(self, recommend_value, test_data_value):
        correct_index = 0
        for value in recommend_value:
            if value in test_data_value:
                correct_index += 1

        return correct_index * 1.0 / len(test_data_value)


    def check_recall(self, recommend_value, test_data_value):
        correct_index = 0
        for value in recommend_value:
            if value in test_data_value:
                correct_index += 1

        return correct_index * 1.0 / len(recommend_value)


    def check_coverage(self):
        pass
        

    def check_popilarity(self, movie_count_frame, recommend_value):
        pass
        

if __name__ == '__main__':
    data_frame = read_csv_file()
    train_data_frame, test_data_frame = train_test_split_data(data_frame)
    movie_similar_result_frame, groupby_user = calculate_inverted_list(train_data_frame)    

    recommend_user_id = random.choice(list(groupby_user.groups.keys()))

    recommend_result_series = recommend(movie_similar_result_frame, groupby_user, 100, recommend_user_id)

    test_data_user_like_movies = test_data_frame[test_data_frame['userId'] == recommend_user_id]['movieId'].unique()
    for recommend_movie in recommend_result_series.keys():
        if recommend_movie in test_data_user_like_movies:
            print('this movieId = {0} is real like movie'.format(recommend_movie))

    check_algorithm = CheckAlgorithm()
    algorithm_recall = check_algorithm.check_recall(recommend_result_series.keys(), test_data_user_like_movies)
    algorithm_precision = check_algorithm.check_precision(recommend_result_series.keys(), test_data_user_like_movies)

    print('algorithm_recall = {0}'.format(algorithm_recall))
    print('algorithm_precision = {0}'.format(algorithm_precision))

