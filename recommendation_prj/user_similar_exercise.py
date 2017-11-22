import pandas as pd
import numpy as np
import os

def read_csv_file():
    header = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = '/home/lichenguang/code/Recommendation_Data/ml-20m'
    file_name = 'ratings.csv'
    data_frame = pd.read_csv(os.path.join(file_path, file_name), sep=',', names=header, skiprows=1)

    return data_frame


def calculate_user_similar(data_frame, user_1_id, user_2_id):
    user_1_frame = data_frame[data_frame['userId'] == user_1_id]
    user_2_frame = data_frame[data_frame['userId'] == user_2_id]

    user_1_rated_movie = user_1_frame['movieId'].unique()
    user_2_rated_movie = user_2_frame['movieId'].unique()

    intersection_result = np.intersect1d(user_1_rated_movie, user_2_rated_movie)
    union_result = np.union1d(user_1_rated_movie, user_2_rated_movie)

    similar_result_version1 = len(intersection_result) * 1.0/ len(union_result)
    print(similar_result_version1)

    similar_result_version2 = len(intersection_result) * 1.0/ np.sqrt(len(user_1_rated_movie) * len(user_2_rated_movie))
    print(similar_result_version2)


if __name__ == '__main__':
    data_frame = read_csv_file()
    user_1_id = 1
    user_2_id = 2

    uniq_users = data_frame['userId'].unique()
    uniq_users = uniq_users[:10]

    for user_id in uniq_users:
        if user_id == 1:
            continue
        user_2_id = user_id
        print('*' * 10)
        print('userid = {0}'.format(user_id))
        print('*' * 10)
        calculate_user_similar(data_frame, user_1_id, user_2_id)
