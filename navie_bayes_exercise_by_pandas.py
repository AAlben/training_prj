import pandas as pd
import numpy as np
import os


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec


def create_vocab_list(dataSet):
    vocab_set = set([])
    for document in dataSet:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, post_words_list):
    basic_vocab = np.zeros(len(vocab_list))
    for basic_word in vocab_list:
        if basic_word in post_words_list:
            basic_vocab[vocab_list.index(basic_word)] += 1
    return basic_vocab


def trainNB0(train_matrix_frame):
    lable = train_matrix_frame['lable']
    p_class_1 = sum(lable) * 1.0 / len(lable)

    class_1_frame = train_matrix_frame[train_matrix_frame['lable'] == 1]
    class_1_frame = class_1_frame.drop('lable', axis=1)
    one_row_result = class_1_frame.sum(axis=0)
    '''
        0     5.0
        1     3.0
        2     3.0
        3     3.0
        4     5.0
        5     3.0
        6     4.0
        7     3.0
        8     4.0
        9     3.0
        10    4.0
        11    3.0
        12    3.0
        13    4.0
        14    3.0
        15    4.0
        16    4.0
        17    4.0
        18    4.0
        19    3.0
        20    3.0
        21    3.0
        22    3.0
        23    4.0
        24    3.0
        25    4.0
        26    4.0
        27    3.0
        28    4.0
        29    6.0
        30    3.0
        31    3.0
        dtype: float64
    '''
    one_column_result = one_row_result.sum() + 2
    one_row_result += 1    
    p1_vocab_frame = np.log(one_row_result / one_column_result)


    class_0_frame = train_matrix_frame[train_matrix_frame['lable'] == 0]
    class_0_frame = class_0_frame.drop('lable', axis=1)
    one_row_result = class_0_frame.sum(axis=0)
    one_column_result = one_row_result.sum() + 2
    one_row_result += 1
    p0_vocab_frame = np.log(one_row_result / one_column_result)
    return p0_vocab_frame, p1_vocab_frame, p_class_1


def trainNB0_by_numpy(train_matrix, train_category):
    '''
    train_matrix = 
    [[0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 1 0 1 0 1 0 0 0]
     [1 1 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1]
     [0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1]
     [0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0]]
    '''

    # train_category = [0 1 0 1 0 1]

    num_train_docs = len(train_matrix)
    # num_train_docs = 6
    num_words = len(train_matrix[0])
    # num_words = 32
    p_abusive = sum(train_category) / float(num_train_docs)
    # p_abusive = 3 / 6

    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    '''
    p1_num = [ 1.  1.  1.  2.  2.  1.  1.  1.  2.  2.  1.  1.  1.  2.  2.  2.  2.  2.
            1.  3.  1.  2.  2.  1.  3.  1.  4.  1.  2.  1.  1.  1.]
    p1_denom = 21

    p1_vect = p1_num / p1_denom
    p0_vect = p0_num / p0_denom

    p0_vect = 
    [ 0.07692308  0.07692308  0.07692308  0.03846154  0.03846154  0.07692308
      0.07692308  0.07692308  0.03846154  0.07692308  0.07692308  0.07692308
      0.07692308  0.03846154  0.03846154  0.11538462  0.03846154  0.03846154
      0.07692308  0.03846154  0.07692308  0.07692308  0.03846154  0.07692308
      0.07692308  0.07692308  0.03846154  0.07692308  0.03846154  0.07692308
      0.07692308  0.15384615]
    p1_vect = 
    [ 0.04761905  0.04761905  0.04761905  0.0952381   0.0952381   0.04761905
      0.04761905  0.04761905  0.0952381   0.0952381   0.04761905  0.04761905
      0.04761905  0.0952381   0.0952381   0.0952381   0.0952381   0.0952381
      0.04761905  0.14285714  0.04761905  0.0952381   0.0952381   0.04761905
      0.14285714  0.04761905  0.19047619  0.04761905  0.0952381   0.04761905
      0.04761905  0.04761905]
    '''

    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    # numpy.log = log e, value

    return p0_vect, p1_vect, p_abusive


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ....]
    classVec = [0, 1, 0, 1, 0, 1] 
    '''
    vocab_list = create_vocab_list(postingList)
    # vocab_list = ['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', ....]

    train_matrix = []

    for posting in postingList:
        return_vec = set_of_words_to_vec(vocab_list, posting)
        train_matrix.append(return_vec)

    train_matrix_frame = pd.DataFrame(train_matrix)
    # train_matrix_frame += 1
    train_matrix_frame['lable'] = classVec

    '''
    train_matrix_frame = 
            26   27   28   29   30   31  lable  
        0  0.0  0.0  0.0  0.0  0.0  0.0      0  
        1  0.0  0.0  0.0  1.0  0.0  1.0      1  
        2  0.0  1.0  0.0  0.0  0.0  0.0      0  
        3  0.0  0.0  1.0  0.0  0.0  0.0      1  
        4  1.0  0.0  0.0  0.0  0.0  0.0      0  
        5  0.0  0.0  0.0  0.0  1.0  0.0      1 
    '''

    p0_vocab_frame, p1_vocab_frame, p_class_1 = trainNB0(train_matrix_frame)

    p0_vect, p1_vect, p_abusive = trainNB0_by_numpy(train_matrix, classVec)

    print(p1_vocab_frame)
    print(p1_vect)
