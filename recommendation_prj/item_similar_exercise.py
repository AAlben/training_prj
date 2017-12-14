import pandas as pd
import numpy as np
from numpy import *
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
    denominator = np.sqrt(denominator)

    movie_similar_diag_0_matrix = 1.0 * movie_similar_diag_0_matrix / denominator
    movie_similar_diag_0_matrix = movie_similar_diag_0_matrix.T / denominator
    movie_similar_diag_0_matrix = movie_similar_diag_0_matrix.T

    max_similar = movie_similar_diag_0_matrix.max(0)
    movie_similar_diag_0_matrix = 1.0 * movie_similar_diag_0_matrix / max_similar
    movie_similar_result_frame = pd.DataFrame(movie_similar_diag_0_matrix, index=data_frame_pivot.columns, columns=data_frame_pivot.columns)

    return movie_similar_result_frame, groupby_user, data_frame_pivot


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
        

def RandomSelectNegativeSample(data_frame, user_rated_movieId_list):
    user_rated_movieId_list = user_rated_movieId_list.values
    dont_rated_data_frame = data_frame[data_frame.apply(lambda x: x['movieId'] not in user_rated_movieId_list, axis=1)]
    sample_frame = dont_rated_data_frame.sample(len(user_rated_movieId_list))

    ret_frame = pd.DataFrame(user_rated_movieId_list, columns=['movieId'])
    ret_frame['rev_value'] = np.ones(len(user_rated_movieId_list))

    ret_frame_other = pd.DataFrame(sample_frame['movieId'], columns=['movieId'])
    ret_frame_other['rev_value'] = np.zeros(sample_frame['movieId'].shape[0])

    ret_frame = pd.concat([ret_frame, ret_frame_other])
    return ret_frame


def sigmoid(ixX):
    return 1 / (1 + np.exp(-inx))


def latent_factor_model(user_item, groupby_user, F, N, alpha, lambda_f, train_data_frame):
    data_frame_pivot = user_item
    P, Q, F = init_model(data_frame_pivot, F)
    # F = 31
    # N = movieId length

    N = 10

    for step in range(N):

        for userId, index_list in groupby_user.groups.items():
            items = groupby_user.get_group(userId)['movieId']
            samples = RandomSelectNegativeSample(train_data_frame, items)

            for iloc_index in range(samples.shape[0]):
                sample_item = samples.iloc[iloc_index]
                movieId = sample_item['movieId']
                rui = sample_item['rev_value']

                eui = rui - np.dot(P.loc[userId], Q.loc[movieId].T)

                for f in range(0, F):
                    P.loc[userId][f] += alpha * (eui * Q.loc[movieId][f] - lambda_f * P.loc[userId][f])
                    Q.loc[movieId][f] += alpha * (eui * P.loc[userId][f] - lambda_f * Q.loc[movieId][f])

            alpha *= 0.9

    print(P)

    return P, Q, F


def init_model(user_item, F):
    data_frame_pivot = user_item.values

    U, sigma, VT = np.linalg.svd(data_frame_pivot)
    sigma_2 = sigma ** 2
    sum_sigma_2 = sum(sigma_2) * 0.9
    sigma_K = 31

    F = sigma_K
    # sigma_F = np.eye(F) * sigma[:F]

    # Q = data_frame_pivot.T * np.mat(U[:, :F]) * sigma_F
    # P = (sigma_F * np.mat(VT[:F]) * data_frame_pivot.T).T

    P = np.ones((U.shape[0], F))
    Q = np.ones((VT.shape[0], F))

    P = pd.DataFrame(P, index=user_item.index, columns=range(F))
    Q = pd.DataFrame(Q, index=user_item.columns, columns=range(F))
    
    return P, Q, F





def Recommend(user, P, Q):
    rank = dict()
    for f, puf in P[user].items():
        for i, qfi in Q[f].items():
            if i not in rank:
                rank[i] += puf * qfi

    return rank


class CalculateSimilar(object):
    def ecludSim(self, inA, inB):
        return 1.0 / (1.0 + np.linalg.norm(inA - inB))


    def pearsSim(self, inA, inB):
        # pearsSim_result = np.corrcoef(inA, inB, rowvar=0)
        '''
        [[ 1.          0.26899111]
         [ 0.26899111  1.        ]]
        '''

        return 0.5 * (1 + np.corrcoef(inA, inB, rowvar=0)[0][1])


    def cosSim(self, inA, inB):
        num = float(np.dot(inA.T, inB))
        denom = np.linalg.norm(inA) * np.linalg.norm(inB)
        return 0.5 * (1 + num / denom)
        


if __name__ == '__main__':
    data_frame = read_csv_file()
    train_data_frame, test_data_frame = train_test_split_data(data_frame)
    movie_similar_result_frame, groupby_user, data_frame_pivot = calculate_inverted_list(train_data_frame)    

    # calculate_similar = CalculateSimilar()
    # print(calculate_similar.ecludSim(data_frame_pivot[3814], data_frame_pivot[315]))
    # print(calculate_similar.pearsSim(data_frame_pivot[3814], data_frame_pivot[315]))
    # print(calculate_similar.cosSim(data_frame_pivot[3814], data_frame_pivot[315]))

    latent_factor_model(data_frame_pivot, groupby_user, None, data_frame_pivot.shape[1], 0.02, 0.01, train_data_frame)

    recommend_user_id = random.choice(list(groupby_user.groups.keys()))

    # ret_frame = RandomSelectNegativeSample(train_data_frame, groupby_user, recommend_user_id)   
    '''
              movieId  rev_value
        1627     1196        1.0
        1794    48982        1.0
        1796    50872        1.0
        1611      953        1.0
        1587      529        1.0
        1777    26294        1.0
        1800    54004        1.0
        1739     5620        1.0
        1696     2804        1.0
        1660     2049        1.0
        1745     6358        1.0
        1715     3723        1.0
        1767     8633        1.0
        1752     6753        1.0
        1578      440        1.0
        1705     3175        1.0
        1675     2193        1.0
        1564       17        1.0
        1744     6297        1.0
        1701     3088        1.0
        1613      955        1.0
        1748     6533        1.0
        1673     2161        1.0
        1716     3751        1.0
        1667     2094        1.0
        1631     1207        1.0
        1677     2324        1.0
        1617     1022        1.0
        1749     6550        1.0
        1645     1441        1.0
        ...
        3897     3814        0.0
        1527      315        0.0
        2553       62        0.0
        399      2750        0.0
        1298     8972        0.0
    '''


    '''

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
    '''
