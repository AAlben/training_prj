import pandas as pd
import numpy as np
import os

from multiprocessing import Process
from multiprocessing.sharedctypes import Value, Array


def read_csv_file():
    header = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = '/home/lichenguang/code/Recommendation_Data/ml-20m'
    file_name = 'ratings.csv'
    data_frame = pd.read_csv(os.path.join(file_path, file_name), sep=',', names=header, skiprows=1)

    data_frame = data_frame.drop(['rating', 'timestamp'], axis=1)
    return data_frame


def calculate_inverted_list(data_frame):
    groupby_movie = data_frame.groupby('movieId')
    groupby_user = data_frame.groupby('userId')
    user_count_frame = groupby_user.count()
    movie_count_frame = groupby_movie.count()

    movie_list = groupby_movie.groups.keys()
    movieId_series = pd.Series(movie_list)
    movieId_series_T = pd.Series(movieId_series.keys(), index=movie_list)

    item_matrix = np.zeros((movie_count_frame.shape[0], movie_count_frame.shape[0]))

    for userId, group_index_list in groupby_user.groups.items():
        p = Process(target=process_calculate, args=(item_matrix, movie_count_frame, group_index_list, movieId_series_T, data_frame))
        p.start()

    print(item_matrix)

def process_calculate(item_matrix, movie_count_frame, group_index_list, movieId_series_T, data_frame):            
        one_row_matrix = np.zeros((1, movie_count_frame.shape[0]))
        one_column_matrix = np.ones((movie_count_frame.shape[0], 1))

        for group_index in group_index_list:
            matrix_index = movieId_series_T[data_frame.iloc[group_index]['movieId']]
            one_row_matrix[0][matrix_index] += 1

        multiplication_matrix = one_row_matrix * one_column_matrix
        multiplication_matrix = np.triu(multiplication_matrix)
        diagonal = np.diag(np.diag(multiplication_matrix))
        half_similar_matrix = multiplication_matrix - diagonal
        half_similar_matrix += half_similar_matrix.T
        item_matrix += half_similar_matrix

def f():    

    for userId, group_index_list in groupby_user.groups.items():
        
        one_row_matrix = np.zeros((1, movie_count_frame.shape[0]))
        one_column_matrix = np.ones((movie_count_frame.shape[0], 1))

        for group_index in group_index_list:
            matrix_index = movieId_series_T[data_frame.iloc[group_index]['movieId']]
            one_row_matrix[0][matrix_index] += 1

        multiplication_matrix = one_row_matrix * one_column_matrix
        multiplication_matrix = np.triu(multiplication_matrix)
        diagonal = np.diag(np.diag(multiplication_matrix))
        half_similar_matrix = multiplication_matrix - diagonal
        half_similar_matrix += half_similar_matrix.T
        item_matrix += half_similar_matrix

    print(item_matrix)

    '''
    item_matrix = 
                173     447     2283    2391    2719    3895    4397    4878
        173        NaN    39.0    67.0   811.0   690.0   181.0   179.0  1841.0   
        447       39.0     NaN    14.0    40.0    29.0    12.0    13.0    24.0   
        2283      67.0    14.0     NaN   134.0    42.0    12.0    10.0    93.0   
        2391     811.0    40.0   134.0     NaN   540.0   164.0    90.0  1141.0   
        2719     690.0    29.0    42.0   540.0     NaN   146.0    80.0   756.0   
        3895     181.0    12.0    12.0   164.0   146.0     NaN    24.0   181.0   
        4397     179.0    13.0    10.0    90.0    80.0    24.0     NaN   147.0   
        4878    1841.0    24.0    93.0  1141.0   756.0   181.0   147.0     NaN 
    '''

    rated_movie_user_count = pd.Series(0, index=movie_list)
    for movieId in rated_movie_user_count.keys():
        if movieId in movie_count_frame.index:
            rated_movie_user_count[movieId] = movie_count_frame['userId'][movieId]

    rated_movie_user_count = np.sqrt(rated_movie_user_count)
    division_result = item_matrix / rated_movie_user_count    
    division_result = division_result.T / rated_movie_user_count
    division_result = division_result.T

    '''
    division_result = 
                  329       1090      1588      1768      1840      2460      2577  
        329          NaN  0.162158  0.088673  0.003958  0.050143  0.048100  0.018795   
        1090    0.162158       NaN  0.120980  0.010210  0.103116  0.084807  0.035328   
        1588    0.088673  0.120980       NaN  0.017144  0.065166  0.055483  0.027916   
        1768    0.003958  0.010210  0.017144       NaN  0.016115  0.014879       NaN   
        1840    0.050143  0.103116  0.065166  0.016115       NaN  0.045200  0.048108   
        2460    0.048100  0.084807  0.055483  0.014879  0.045200       NaN  0.015143   
        2577    0.018795  0.035328  0.027916       NaN  0.048108  0.015143       NaN   
        2987    0.185868  0.279868  0.151397  0.013088  0.097185  0.080947  0.041439   
        3234    0.004101  0.005291  0.010661       NaN       NaN       NaN       NaN   
        3977    0.139609  0.210938  0.141456  0.015333  0.071128  0.060283  0.032596   
    '''

    print(division_result)

    import pickle
    pickle.dump(division_result, open('division_result.p', 'wb'))

    return division_result, groupby_user, movie_list, user_count_frame


def recommend(item_similar_matrix, groupby_user, recommend_item_K, recommend_movie_id, recommend_user_id):
    with_other_item_similar = item_similar_matrix[recommend_movie_id]

    sorted_similar = with_other_item_similar.sort_values(ascending=False)

    K = 10

    sorted_similar = sorted_similar.iloc[:K]

    recommend_result = {}

    for similar_movieId in sorted_similar.keys():
        user_rated_movieId_list = groupby_user.get_group(recommend_user_id)['movieId']

        if similar_movieId in user_rated_movieId_list:
            recommend_result[similar_movieId] = recommend_result.get(similar_movieId, 0)
            recommend_result[similar_movieId] += item_similar_matrix[recommend_movie_id][similar_movieId]

    recommend_result = pd.Series(recommend_result)
    recommend_result = recommend_result.sort_values(ascending=False)
    
    print(recommend_result)

    return recommend_result.iloc[:recommend_item_K]


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

    division_result, groupby_user, movie_list, user_count_frame = calculate_inverted_list(train_data_frame)

    # test_movie_id = movie_list[0]
    # import random
    # test_user_id_list = random.sample(groupby_user.groups.keys(), 50)

    # for test_user_id in test_user_id_list:
    #     recommend_result = recommend(division_result, groupby_user, 10, test_movie_id, test_user_id)
    #     print(recommend_result)
    '''
    recommend_result = 
        35227     0.968211
        136268    0.847327
        60159     0.837163
        3907      0.831285
        73611     0.830864
        31181     0.812670
        51703     0.774685
        103842    0.754271
        23599     0.745975
        34576     0.744168
        dtype: float64    
    '''

    '''
    test_data_user_like_movies = test_data_frame[test_data_frame['userId'] == test_user_id]['movieId'].unique()

    print(recommend_result)

    for recommend_movie in recommend_result.keys():
        if recommend_movie in test_data_user_like_movies:
            print('this movieId = {0} is real like movie'.format(recommend_movie))

    check_algorithm = CheckAlgorithm()
    algorithm_recall = check_algorithm.check_recall(recommend_result.keys(), test_data_user_like_movies)
    algorithm_precision = check_algorithm.check_precision(recommend_result.keys(), test_data_user_like_movies)

    print('algorithm_recall = {0}'.format(algorithm_recall))
    print('algorithm_precision = {0}'.format(algorithm_precision))

    '''
