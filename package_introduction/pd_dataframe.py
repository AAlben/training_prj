import pandas as pd
import numpy as np

# execute python online url = https://www.tutorialspoint.com/execute_python_online.php

def pandas_exercise():
    set_index = range(1, 10, 3)

    df = pd.DataFrame(None, index=set_index, columns=['user_1', 'user_2'])

    # 定位
    print(df.loc[4]) # 值定位
    # df.ix - index
    # df.iloc - index

    # 整理列，去掉不用的
    # df = df[['user_1']]
    # df = df[['user_2']]
    # df = df[['user_1, user_2']]

    # 赋值
    for index in df.index:
        df.at[index, 'user_1'] = index * index
        df.at[index, 'user_2'] = index * 2

    # 创建新列，并赋值
    ret_frame['rev_value'] = np.ones(len(user_rated_movieId_list))

    # 值，列，index    
    for column in df.keys():
        print(column)
    # df.values
    # df.columns
    # df.index


def pandas_groupby_exercise():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
               
    df = pd.DataFrame(dataSet, columns=['v1', 'v2', 'label'])

    print(df)
    print('-' * 10)

    groupby_result = df.groupby('label')

    for key, val in groupby_result.groups.items():
        for v in val:
            # print(df.iloc[v])
            pass


    n_a = np.array([0, 1, 0, 1])

    print(n_a[2])

    n_b = np.ones((n_a.shape[0], 1))
    multiplication_result = n_a * n_b
    print('-' * 10)

    multiplication_result = np.triu(multiplication_result)
    diagonal = np.diag(np.diag(multiplication_result))

    half_item_similar = multiplication_result - diagonal

    half_item_similar += half_item_similar.T
    print(half_item_similar)
    print('-' * 10)

    value = range(1, 10, 2)
    movieId_series = pd.Series(value)
    movieId_series_T = pd.Series(movieId_series.keys(), index=value)

    print(movieId_series)
    print(movieId_series_T)

    print('-' * 10)

    one_row_matrix = np.zeros((1, 25590))
    matrix_index = 1587
    one_row_matrix[0][matrix_index] += 1
    print(one_row_matrix)