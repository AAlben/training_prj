import numpy as np
import pandas as pd
import time
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

import os

def read_file():
    header = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = '/home/lichenguang/code/Recommendation_Data/ml-20m'
    file_name = 'ratings.csv'
    data_frame = pd.read_csv(os.path.join(file_path, file_name), sep=',', names=header, skiprows=1)

    data_frame = data_frame[data_frame['userId'] < 1000]
    data_frame = data_frame[data_frame['movieId'] < 1000]
    # data_frame = data_frame.iloc[0:1000, :]
    print(data_frame)

    users_unique_count = data_frame.userId.unique().shape[0]
    items_unique_count = data_frame.movieId.unique().shape[0]

    print('all user is {0} ; all item is {1}'.format(users_unique_count, items_unique_count))

    train_data, test_data = ms.train_test_split(data_frame, test_size=0.25)

    train_data_matrix = np.zeros((1000, 1000))
    for data_line in train_data.itertuples():
        # data_line = Pandas(Index=724, userId=7, movieId=3086, rating=4.0, timestamp=1011205452)
        train_data_matrix[data_line[1] - 1, data_line[2] - 1] = data_line[3]

    test_data_matrix = np.zeros((1000, 1000))
    for data_line in test_data.itertuples():
        test_data_matrix[data_line[1] - 1, data_line[2] - 1] = data_line[3]

    user_similar = pairwise_distances(train_data_matrix, metric='cosine')
    item_similar = pairwise_distances(train_data_matrix.T, metric='cosine')

    return (train_data_matrix, test_data_matrix, user_similar, item_similar)


def predict(rating, similar, type='user'):
    if type == 'user':
        mean_user_rating = rating.mean(axis=1)
        rating_diff = (rating - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similar.dot(rating_diff) / np.array([np.abs(similar).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similar) / np.array([np.abs(similar).sum(axis=1)])

    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


if __name__ == '__main__':
    train_data_matrix, test_data_matrix, user_similar, item_similar = read_file()

    user_prediction = predict(train_data_matrix, user_similar, type='user')
    item_prediction = predict(train_data_matrix, item_similar, type='item')

    print(rmse(user_prediction, test_data_matrix))
    print(rmse(item_prediction, test_data_matrix))










