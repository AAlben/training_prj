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


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    # return_vec = [0, 0, 0, 0, 0, 0... ]

    for word in input_set:
        # input_set = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: {0} is not in my Vocabulary!'.format(word))
    return return_vec


def bag_of_words_to_vec_MN(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    # return_vec = [0, 0, 0, 0, 0, 0... ]

    for word in input_set:
        # input_set = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def i_think_correct_trainNB0(train_matrix, train_category):
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

    p0_num = np.zeros(num_words)
    p1_num = np.zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    '''
    p1_num = [ 0.  0.  0.  1.  1.  0.  0.  0.  1.  1.  0.  0.  0.  1.  1.  1.  1.  1.
              0.  2.  0.  1.  1.  0.  2.  0.  3.  0.  1.  0.  0.  0.]
    p1_denom = 19.0
    '''

    p1_vect = p1_num / p1_denom
    p0_vect = p0_num / p0_denom
    '''
    p0_vect = 
    [ 0.04166667  0.04166667  0.04166667  0.          0.          0.04166667
      0.04166667  0.04166667  0.          0.04166667  0.04166667  0.04166667
      0.04166667  0.          0.          0.08333333  0.          0.
      0.04166667  0.          0.04166667  0.04166667  0.          0.04166667
      0.04166667  0.04166667  0.          0.04166667  0.          0.04166667
      0.04166667  0.125     ]
    p1_vect = 
    [ 0.          0.          0.          0.05263158  0.05263158  0.          0.
      0.          0.05263158  0.05263158  0.          0.          0.
      0.05263158  0.05263158  0.05263158  0.05263158  0.05263158  0.
      0.10526316  0.          0.05263158  0.05263158  0.          0.10526316
      0.          0.15789474  0.          0.05263158  0.          0.          0.        ]
    '''

    return p0_vect, p1_vect, p_abusive


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


def classify_NB(vec_to_classify, p0_vect, p1_vect, p_class1):
    # p_class1 = p_abusive = 0.5
    # log(a * b) = log(a) + log(b)
    p1 = sum(vec_to_classify * p1_vect) + np.log(p_class1)
    p0 = sum(vec_to_classify * p0_vect) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


class ClassifySpam(object):
    def __init__(self):
        pass

    def textParse(self, content):
        import re
        content = unicode(content, errors='ignore')
        list_of_tokens = re.split(r'\W*', content)
        return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


    def spam_test(self):
        import random
        doc_list = []
        class_list = []
        full_text = []

        email_file_path = '/home/lichenguang/code/Bayes_Data'

        for i in range(1, 26):
            # word_list = self.textParse(open(os.path.join(email_file_path, 'email/spam/{0}.txt'.format(i))).read())

            file_name = os.path.join(email_file_path, 'email/spam/{0}.txt'.format(i))
            file_obj = open(file_name, 'r')
            word_list = self.textParse(file_obj.read())

            doc_list.append(word_list)
            full_text.append(word_list)
            class_list.append(1)

            # word_list = self.textParse(open(os.path.join(email_file_path, 'email/ham/{0}.txt'.format(i))).read())

            file_name = os.path.join(email_file_path, 'email/ham/{0}.txt'.format(i))
            file_obj = open(file_name, 'r')
            word_list = self.textParse(file_obj.read())

            doc_list.append(word_list)
            full_text.append(word_list)
            class_list.append(0)

        vocab_list = create_vocab_list(doc_list)
        training_set = range(50)
        test_set = []

        for i in range(10):
            rand_index = int(random.uniform(0, len(training_set)))
            test_set.append(training_set[rand_index])
            del(training_set[rand_index])

        train_mat = []
        train_classes = []

        print(test_set)

        for doc_index in training_set:
            train_mat.append(bag_of_words_to_vec_MN(vocab_list, doc_list[doc_index]))
        pass


if __name__ == '__main__':
    classify_words_flag = False
    classify_file_flag = True

    if classify_words_flag:
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

        p0_vect, p1_vect, p_abusive = trainNB0(np.array(train_matrix), np.array(classVec))

        '''
        p0_vect = 
        [-2.56494936 -2.56494936 -2.56494936 -3.25809654 -3.25809654 -2.56494936
         -2.56494936 -2.56494936 -3.25809654 -2.56494936 -2.56494936 -2.56494936
         -2.56494936 -3.25809654 -3.25809654 -2.15948425 -3.25809654 -3.25809654
         -2.56494936 -3.25809654 -2.56494936 -2.56494936 -3.25809654 -2.56494936
         -2.56494936 -2.56494936 -3.25809654 -2.56494936 -3.25809654 -2.56494936
         -2.56494936 -1.87180218]
        p1_vect = 
        [-3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244
         -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -3.04452244
         -3.04452244 -2.35137526 -2.35137526 -2.35137526 -2.35137526 -2.35137526
         -3.04452244 -1.94591015 -3.04452244 -2.35137526 -2.35137526 -3.04452244
         -1.94591015 -3.04452244 -1.65822808 -3.04452244 -2.35137526 -3.04452244
         -3.04452244 -3.04452244]   
        '''

        test_entry_list = [
                            ['love', 'my', 'dalmation'],
                            ['stupid', 'garbage']
        ]

        for test_entry in test_entry_list:
            return_vec = set_of_words_to_vec(vocab_list, test_entry)
            classified_result = classify_NB(return_vec, p0_vect, p1_vect, p_abusive)
            print(classified_result)

    if classify_file_flag:
        classify_spam = ClassifySpam()
        classify_spam.spam_test()