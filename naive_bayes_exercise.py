import numpy as np


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


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    # return_vec = [0, 0, 0, 0, 0, 0... ]

    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: {0} is not in my Vocabulary!'.format(word))
    return return_vec


def trainNB0(train_matrix, train_category):
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

    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    # numpy.log = log e, value

    return p0_vect, p1_vect, p_abusive


def classify_NB(vec_to_classify, p0_vect, p1_vect, p_class1):
    p1 = sum(vec_to_classify * p1_vect) + np.log(p_class1)
    p0 = sum(vec_to_classify * p0_vect) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    vocab_list = create_vocab_list(postingList)

    train_matrix = []

    for posting in postingList:
        return_vec = set_of_words_to_vec(vocab_list, posting)
        train_matrix.append(return_vec)

    p0_vect, p1_vect, p_abusive = trainNB0(np.array(train_matrix), np.array(classVec))

    test_entry = ['love', 'my', 'dalmation']
    return_vec = set_of_words_to_vec(vocab_list, test_entry)
    classified_result = classify_NB(return_vec, p0_vect, p1_vect, p_abusive)
    print(classified_result)