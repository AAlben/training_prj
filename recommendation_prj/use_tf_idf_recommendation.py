

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfTransformer


def get_news_file():
    import os

    news_list = []
    project_path = os.getcwd()
    for file_item in os.listdir(project_path):
        if '.txt' in file_item:            
            file_name = file_item
            news_content = None
            with open(file_name, 'r') as fp:
                news_content = fp.readlines()
            news_content = ''.join(news_content)

            news_list.append(news_content)
    return news_list


def init_news_time(news_count):
    import copy

    import_time_list = []
    publish_time_list = []

    now_time = int(time.time())
    step_size = 1000

    for i in range(news_count):
        import_time_list.append(now_time - step_size * i)

    publish_time_list = copy.copy(import_time_list)
    publish_time_list.reverse()
    
    return import_time_list, publish_time_list


def cut_sentence(news_content, all_news_word_set):
    import jieba.analyse

    words_count_matrix = {}
    tags_with_weight_list = jieba.analyse.extract_tags(news_content, topK=100, withWeight=True)    

    for tag_with_weight in tags_with_weight_list:
        if tag_with_weight[0] not in all_news_word_set:
            all_news_word_set.add(tag_with_weight[0])

        words_count_matrix[tag_with_weight[0]] = tag_with_weight[1]

    return pd.Series(words_count_matrix)


def init_keywords():
    news_list = get_news_file()
    pass


def init_now_news(now_news_key):
    # import jieba.analyse
    # read new news from mysql

    news_list = get_news_file()
    data_frame = pd.DataFrame(news_list, columns=['content'])

    all_news_word_set = set()    

    data_frame = data_frame.applymap(lambda x: cut_sentence(x, all_news_word_set))
    words_count_frame = pd.concat(data_frame['content'].values, axis=1)
    words_count_frame = words_count_frame.T
    words_count_frame = words_count_frame.fillna(0)

    import_time_list, publish_time_list = init_news_time(len(news_list))

    data_frame['import_time'] = pd.Series(import_time_list)
    data_frame['publish_time'] = pd.Series(publish_time_list)
    data_frame['news_id'] = pd.Series(range(len(news_list)))

    ''' serealize to redis + unserealize from redis
    # redis_client.set(now_news_key, data_frame.to_msgpack(compress='zlib'))
    # data_frame = pd.read_msgpack(redis_client.get(now_news_key))
    '''

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(words_count_frame.as_matrix())   
    weight = tfidf_matrix.toarray()

    '''
    weight = 
    [[ 0.          0.          0.          0.57735027  0.57735027  0.57735027]
     [ 0.          0.          0.63720629  0.44495998  0.44495998  0.44495998]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.47336192  0.47336192  0.47336192  0.33054776  0.33054776  0.33054776]
     [ 0.65015778  0.65015778  0.          0.22700199  0.22700199  0.22700199]]
    weight.shape = (5, 4083)
    '''
    
    # print(weight.shape)
    news_similar = np.dot(weight, weight.T)
    # print(news_similar)

    return data_frame, all_news_word_set, weight


def push_new_news(now_news_key, news_data_frame, all_news_word_set):
    import jieba.analyse

    file_name = 'news.txt'
    with open(file_name, 'r') as fp:
        news_content = fp.readlines()
    news_content = ''.join(news_content)
    tags_with_weight_list = jieba.analyse.extract_tags(news_content, topK=10, withWeight=True)

    words_count_matrix = {}

    for tag_with_weight in tags_with_weight_list:
        if tag_with_weight[0] not in all_news_word_set:
            all_news_word_set.add(tag_with_weight[0])

        words_count_matrix[tag_with_weight[0]] = tag_with_weight[1]

    now_time = int(time.time())

    news_item = {'content': pd.Series(words_count_matrix), 'import_time': now_time, 'publish_time': now_time, 'news_id': news_data_frame.shape[0]}

    news_data_frame = news_data_frame.append(pd.Series(news_item), ignore_index=True)

    # print(news_data_frame)

    return news_data_frame, all_news_word_set


def user_click_after(user_id, news_id, now_news_word, news_data_frame, user_feature):
    import re

    news_item = news_data_frame[news_data_frame['news_id'] == news_id]
    news_item_word = news_item['content']

    user_feature_matrix = user_feature.get(user_id, np.zeros((1, len(now_news_word))))

    for word in now_news_word:
        findall_result = re.findall(word, news_item_word.values[0])
        user_feature_matrix[0][now_news_word.index(word)] += len(findall_result)

    user_feature[user_id] = np.divide(user_feature_matrix, len(news_item_word.values[0].split(',')) * 1.0)

    print(user_feature)
    return user_feature[user_id]


def extract_news_feature(words_count_frame): 
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(words_count_frame.as_matrix())   
    weight = tfidf_matrix.toarray()

    return weight


def recommendation_user(user_id, user_feature, news_data_frame, now_news_weight):
    one_user_feature = user_feature[user_id]

    
    for index in range(len(now_news_weight[6])):
        if one_user_feature[0][index] != now_news_weight[6][index]:
            print(index)




    pass



if __name__ == '__main__':
    now_news_key = 'This is Now News Redis Key'
    now_news_word = None

    news_data_frame, all_news_word_set, now_news_weight = init_now_news(now_news_key)

    news_data_frame, all_news_word_set = push_new_news(now_news_key, news_data_frame, all_news_word_set)

    # print(news_data_frame)

    words_count_frame = pd.concat(news_data_frame['content'].values, axis=1)
    words_count_frame = words_count_frame.T
    words_count_frame = words_count_frame.fillna(0)

    # print(words_count_frame)

    now_news_weight = extract_news_feature(news_data_frame)

    user_id = 1
    news_id = 6
    user_feature = {}
    user_feature[user_id] = user_click_after(user_id, news_id, now_news_word, news_data_frame, user_feature)

    recommendation_user(user_id, user_feature, news_data_frame, now_news_weight)
