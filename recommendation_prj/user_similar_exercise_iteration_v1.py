import pandas as pd
import numpy as np
import os

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
    user_count = user_count_frame.shape[0]

    user_count = 10

    user_matrix = {}

    for movieId, rated_users in groupby_movie.groups.items():
        # movieId = 131072
        # rated_users = Int64Index([11528440], dtype='int64')
        #           userId  movieId
        # 11528440   79570   131072

        uniq_this_movie_users = groupby_movie.get_group(movieId)['userId']

        for userId in uniq_this_movie_users:
            if userId >= user_count:
                continue

            for other_userId in uniq_this_movie_users:
                if other_userId >= user_count:
                    continue
                if userId == other_userId:
                    continue

                user_matrix[userId] = user_matrix.get(userId, {})
                user_matrix[userId][other_userId] = user_matrix[userId].get(other_userId, 0)
                user_matrix[userId][other_userId] += 1

    user_matrix = pd.DataFrame(user_matrix)
    '''
    user_matrix = 
              1     2     3     4     5     6     7     8     9     10
        1    NaN   9.0  42.0   3.0  14.0   2.0  21.0   6.0   4.0  11.0
        2    9.0   NaN  15.0   2.0   7.0   3.0  16.0   5.0   NaN   4.0
        3   42.0  15.0   NaN   5.0  16.0   5.0  44.0  10.0   4.0  21.0
        4    3.0   2.0   5.0   NaN  12.0   1.0   5.0  14.0   1.0   1.0
        5   14.0   7.0  16.0  12.0   NaN  11.0  19.0  20.0   NaN   6.0
        6    2.0   3.0   5.0   1.0  11.0   NaN   8.0   4.0   NaN   2.0
        7   21.0  16.0  44.0   5.0  19.0   8.0   NaN  16.0   7.0  15.0
        8    6.0   5.0  10.0  14.0  20.0   4.0  16.0   NaN   1.0   3.0
        9    4.0   NaN   4.0   1.0   NaN   NaN   7.0   1.0   NaN   2.0
        10  11.0   4.0  21.0   1.0   6.0   2.0  15.0   3.0   2.0   NaN
    '''

    user_rated_movie_count = pd.Series(0, index=range(1, user_count + 1))
    for userId in range(1, user_count + 1):
        if userId in user_count_frame.index:
            user_rated_movie_count[userId] = user_count_frame['movieId'][userId]

    user_rated_movie_count = np.sqrt(user_rated_movie_count)
    division_result = user_matrix / user_rated_movie_count    
    division_result = division_result.T / user_rated_movie_count
    division_result = division_result.T

    '''
    division_result = 
                  1         2         3         4         5         6         7   \
        1        NaN  0.087108  0.232172  0.042857  0.130268  0.030861  0.095553   
        2   0.087108       NaN  0.140445  0.048393  0.110322  0.078406  0.123311   
        3   0.232172  0.140445       NaN  0.069099  0.144021  0.074635  0.193677   
        4   0.042857  0.048393  0.069099       NaN  0.279145  0.038576  0.056877   
        5   0.130268  0.110322  0.144021  0.279145       NaN  0.276385  0.140776   
        6   0.030861  0.078406  0.074635  0.038576  0.276385       NaN  0.098295   
        7   0.095553  0.123311  0.193677  0.056877  0.140776  0.098295       NaN   
        8   0.054210  0.076517  0.087404  0.316228  0.294245  0.097590  0.115111   
        9   0.051110       NaN  0.049443  0.031944       NaN       NaN  0.071221   
        10       NaN       NaN       NaN       NaN       NaN       NaN       NaN   
    '''
    return division_result, groupby_user


def recommend(user_similar_matrix, groupby_user, recommend_item_K, recommend_user_id):
    with_other_user_similar = user_similar_matrix[recommend_user_id]

    sorted_similar = with_other_user_similar.sort_values(ascending=False)

    K = 10

    sorted_similar = sorted_similar.iloc[:K]

    recommend_result = {}

    for similar_userId in sorted_similar.keys():
        similar_user_rated_movie = groupby_user.get_group(similar_userId)['movieId']
        for movieId in similar_user_rated_movie:
            recommend_result[movieId] = recommend_result.get(movieId, 0)
            recommend_result[movieId] += user_similar_matrix[recommend_user_id][similar_userId]

    recommend_result = pd.Series(recommend_result)
    recommend_result = recommend_result.sort_values(ascending=False)
    
    return recommend_result.iloc[:recommend_item_K]


def train_test_split_data(data_frame):
    from sklearn import model_selection
    train_data_frame, test_data_frame =  model_selection.train_test_split(data_frame, test_size=0.25)
    return train_data_frame, test_data_frame


if __name__ == '__main__':
    data_frame = read_csv_file()
    train_data_frame, test_data_frame = train_test_split_data(data_frame)

    division_result, groupby_user = calculate_inverted_list(train_data_frame)

    test_user_id = 1
    test_data_user_like_movies = test_data_frame[test_data_frame['userId'] == test_user_id]['movieId'].unique()

    recommend_result = recommend(division_result, groupby_user, len(test_data_user_like_movies), test_user_id)
    print(recommend_result)

    for recommend_movie in recommend_result.keys():
        if recommend_movie in test_data_user_like_movies:
            print('this movieId = {0} is real like movie'.format(recommend_movie))
