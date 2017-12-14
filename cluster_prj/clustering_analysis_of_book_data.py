import pandas as pd
import numpy as np

import os


def load_data_file(file_path, file_name, encoding=None):
    file_name = os.path.join(file_path, file_name)
    data_frame = pd.read_csv(file_name, sep=';', encoding=encoding, error_bad_lines=False)
    return data_frame
    

def structuring_data(book_rating_frame, user_frame, book_frame=None):
    join_frame = pd.merge(book_rating_frame, user_frame, on='User-ID')
    join_frame = join_frame.drop(['Location'], axis=1)

    book_frame = book_frame.drop(['Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
    join_frame = pd.merge(join_frame, book_frame, on='ISBN')

    join_frame = join_frame.iloc[0:5000]

    groupby_age_frame = join_frame.groupby('Age').count().sort_values(by='User-ID', ascending=False)    
    groupby_ISBN_count_frame = join_frame.groupby(['ISBN', 'Book-Title']).count().drop(['User-ID', 'Book-Rating'], axis=1).rename({'Age': 'ISBN_count'}, axis=1)

    for age_index in range(10):
        start_age = age_index * 10 + 1
        end_age = (age_index + 1) * 10

        age_30_frame = join_frame[join_frame['Age'] >= start_age]
        age_30_frame = age_30_frame[age_30_frame['Age'] < end_age]
        groupby_ISBN_frame = age_30_frame.groupby(['ISBN', 'Book-Title']).count().sort_values(by='Age', ascending=False)
        groupby_ISBN_frame = groupby_ISBN_frame.drop(['User-ID', 'Book-Rating'], axis=1).rename({'Age': 'ISBN_count_{0}'.format(start_age)}, axis=1)

        groupby_ISBN_count_frame['ISBN_count_{0}'.format(start_age)] = groupby_ISBN_frame['ISBN_count_{0}'.format(start_age)]

    groupby_ISBN_count_frame = groupby_ISBN_count_frame.fillna(0, axis=1)
    age_with_item_result = groupby_ISBN_count_frame.T / groupby_ISBN_count_frame['ISBN_count']
    age_with_item_result = age_with_item_result.T

    for age_index in range(10):
        start_age = age_index * 10 + 1

        column_name = 'ISBN_count_{0}'.format(start_age)
        print('This age is {0}'.format(start_age))
        print(age_with_item_result.sort_values(by=column_name, axis=0, ascending=False).iloc[:5])


def clustering_data(join_frame):
    pass


if __name__ == '__main__':
    file_path = '/home/lichenguang/code/BookCrossing_Data'
    file_name_list = ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX-Books.csv']

    book_rating_frame = load_data_file(file_path, file_name_list[0], 'ISO-8859-1')
    user_frame = load_data_file(file_path, file_name_list[1], 'ISO-8859-1')
    book_frame = load_data_file(file_path, file_name_list[2], 'ISO-8859-1')

    structuring_data(book_rating_frame, user_frame, book_frame)