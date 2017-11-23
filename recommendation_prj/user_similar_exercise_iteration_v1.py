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
    groupby_movie = data_frame.groupby('movieId', axis=1)

    print(groupby_movie)


if __name__ == '__main__':
    data_frame = read_csv_file()
    calculate_inverted_list(data_frame)
