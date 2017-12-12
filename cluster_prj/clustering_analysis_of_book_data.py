import pandas as pd
import numpy as np

import os


def load_data_file(file_path, file_name):
    file_name = os.path.join(file_path, file_name)
    data_frame = pd.read_csv(file_name, sep=',', header=None)
    return data_frame


if __name__ == '_main__':
    file_path = '/home/lichenguang/code/BookCrossing_Data'
    file_name_list = ['BX-Book-Ratings.csv', 'BX-Users.csv']

    book_rating_frame = load_data_file(file_path, file_name_list[0])
    user_frame = load_data_file(file_path, file_name_list[1])


    join_frame = pd.concat([book_rating_frame, user_frame], axis=1, join='inner')

    print(join_frame)