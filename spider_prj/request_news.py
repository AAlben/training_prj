#!/usr/bin/python
# -*- coding:utf-8 -*-  

import csv
import pandas as pd
import os


def request_sogou_news(url):
    request_flag = True

    if request_flag:
        import requests

        r = requests.get(url)
        html_text = r.text

        with open('sogou_news_home.html', 'w') as fp:
            fp.write(html_text)

    html_text = None
    with open('sogou_news_home.html', 'r') as fp:
        html_text = fp.read() 

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, 'lxml')
    a_document_list = soup.find_all('a')

    with open('list_news_url_0105.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')

        for a_document in a_document_list:
            csv_writer.writerow([a_document.get('href'), a_document.get('title')])



def read_news_url_list(file_name):
    url_list_frame = pd.read_csv(file_name, sep='\t', prefix='X', header=None)
    url_list_frame = url_list_frame.rename(columns={'X0': 'url', 'X1': 'title'})
    url_list_frame = url_list_frame.dropna(axis=0, how='any')

    map_result_frame = url_list_frame.applymap(lambda x: len(x))
    map_result_frame = map_result_frame.rename(columns={'url': 'url_len', 'title': 'title_len'})
    url_list_frame = pd.merge(url_list_frame, map_result_frame, left_index=True, right_index=True)
    
    url_list_frame = url_list_frame[url_list_frame['title_len'] > 8]
    url_list_frame.to_csv(file_name, sep='\t', encoding='utf-8', header=False)



def read_correct_news_url_list(file_name):
    url_list_frame = pd.read_csv(file_name, sep='\t', prefix='X', header=None)
    url_list_frame = url_list_frame.rename(columns={'X0': 'id', 'X1': 'url', 'X2': 'title', 'X3': 'url_len', 'X4': 'title_len'})
    url_list_frame = url_list_frame.drop(['id'], axis=1)

    from newspaper import Article
    import hashlib

    frame_dic = {}

    for item in url_list_frame.iterrows():
        row = item[1]

        hash_val = hashlib.md5(row['url'].encode('utf-8')).hexdigest()

        try:
            article = Article(row['url'], language='zh')
            article.download()
            article.parse()
            news_content = article.text

            if news_content:
                news_content_file_name = '{0}.txt'.format(hash_val)
                url_list_frame.at[item[0], 'hash_id'] = news_content_file_name
                
                with open(news_content_file_name, 'w') as fp:
                    fp.write(news_content)
        except Exception as e:
            print(e)

    url_list_frame.to_csv(file_name, sep='\t', encoding='utf-8', header=False)



def news_content_to_vector():
    import word2vec
    import gensim
    import os

    file_name = 'ff5d1459a1aa1286b16760e6a508882f.txt'
    news_content = None
    with open(file_name, 'r') as fp:
        news_content = fp.read()

    model = gensim.models.Word2Vec(news_content, size=50, window=5, min_count=5, workers=4)

    print(model)

    # result = word2vec.word2vec(os.getcwd(), os.path.join(os.getcwd(), 'text.txt'), binary=0, size=10, verbose=False)
    # print(result)
    pass



def cut_all_news_document():
    import os
    import jieba
    import jieba.analyse
    import re

    stop_words_list = []
    stop_words_file = 'stop_words.txt'

    prog = re.compile(r'\d')    

    # with open(stop_words_file, 'r') as fp:
    #     line_content = fp.readline()
    #     while line_content:            
    #         stop_words_list.append(line_content.strip())
    #         line_content = fp.readline()

    # with open(stop_words_file, 'w') as fp:
    #     for item in stop_words_list:
    #         fp.writelines(item + '\n')

    with open(stop_words_file, 'r') as fp:
        stop_words_list.extend(fp.readlines())

    jieba.analyse.set_stop_words(os.path.join(os.getcwd(), stop_words_file))
    topK = 10

    cuted_word_from_documents = [{}, {}]
    # cuted_word_from_document = {}
    # cuted_result_files = 'frame_cuted_news_content_words.csv'
    cuted_result_files = ['frame_cuted_news_content_words_train.csv', 'frame_cuted_news_content_words_test.csv']

    index = 0

    for file_item in os.listdir():
        train_test_flag_index = None
        if '.txt' in file_item and file_item != 'stop_words.txt':            
            train_test_flag_index = index % 2 

            index += 1
            news_hash_id = file_item.split('.')[0]
            content = None

            with open(file_item, 'r') as fp:
                content = fp.read()

            tags = list(jieba.cut(content))
            tags = list(filter(lambda x: x not in stop_words_list and not prog.match(x), tags))
            cuted_word_from_documents[train_test_flag_index][news_hash_id] = {'hash_id': news_hash_id, 'words': tags}


    for cuted_result_file in cuted_result_files:
        index = cuted_result_files.index(cuted_result_file)    

        cuted_result_frame = pd.DataFrame(cuted_word_from_documents[index])
        cuted_result_frame = cuted_result_frame.T
        cuted_result_frame.to_csv(cuted_result_file, sep='\t', encoding='utf-8', header=False)



def use_gensim_package(file_name):
    cuted_result_frame = pd.read_csv(file_name, sep='\t', prefix='X', header=None)
    cuted_result_frame = cuted_result_frame.rename(columns={'X0': 'id', 'X1': 'hash_id', 'X2': 'words'})
    cuted_result_frame = cuted_result_frame.drop(['id'], axis=1)

    from gensim import corpora, models
    import os

    processed_corpus = cuted_result_frame['words'].map(lambda x: eval(x))
    # dictionary = corpora.Dictionary(list(processed_corpus))

    # print(processed_corpus)
    word2vec_model = models.Word2Vec(sentences=list(processed_corpus),
                 size=200,
                 window=10,
                 min_count=10,
                 workers=4)


    word2vec_model.wv.save_word2vec_format(fname=os.path.join(os.getcwd(), 'corpus.bin'), binary=True)



def load_corpus_and_process(corpus_file):
    from gensim import models
    import os

    model = models.KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), corpus_file), binary=True, unicode_errors='ignore')
    print(len(model.vocab))    
    print(model['特朗普'].shape)
    print(model.similarity('特朗普', '美国'))
    print(model.most_similar(positive=['特朗普'], topn=10))

    pass



def read_news_file(file_name, test_doc_content):
    import gensim
    stop_words_list = []
    stop_words_file = 'stop_words.txt'
    with open(stop_words_file, 'r') as fp:
        stop_words_list.extend(fp.readlines())

    cuted_result_frame = pd.read_csv(file_name, sep='\t', prefix='X', header=None)
    cuted_result_frame = cuted_result_frame.rename(columns={'X0': 'id', 'X1': 'hash_id', 'X2': 'words'})
    cuted_result_frame = cuted_result_frame.drop(['id'], axis=1)

    cuted_result_test_frame = pd.read_csv('frame_cuted_news_content_words_test.csv', sep='\t', prefix='X', header=None)
    cuted_result_test_frame = cuted_result_test_frame.rename(columns={'X0': 'id', 'X1': 'hash_id', 'X2': 'words'})
    cuted_result_test_frame = cuted_result_test_frame.drop(['id'], axis=1)

    test_doc_content.append(cuted_result_frame.values[0])
    test_doc_content.append(cuted_result_test_frame['words'][0])

    processed_corpus = cuted_result_frame['words'].map(lambda x: eval(x))
    for item in processed_corpus.items():
        words_set = set(item[1])
        correct_words = []
        for word in words_set:
            if word not in stop_words_list and len(word) > 1 :
                correct_words.append(word)

        yield gensim.models.doc2vec.TaggedDocument(correct_words, [item[0]])



def use_gensim_doc2vec_package(train_corpus, test_doc_content):
    import gensim
    from gensim import models 
    import os
    import collections

    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=len(train_corpus))
    model.build_vocab(train_corpus)

    time = model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    print(model.infer_vector(test_doc_content[0]))
    print(model.infer_vector(test_doc_content[1]))


    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        if rank != 0:
            print(train_corpus[doc_id].words)
            print(train_corpus[sims[0][0]].words)

            print('*' * 10)
        
        second_ranks.append(sims[1])

    print(collections.Counter(ranks))



def use_gensim_tfidf_package(file_name):
    import gensim
    from gensim import corpora
    stop_words_list = []
    stop_words_file = 'stop_words.txt'
    with open(stop_words_file, 'r') as fp:
        stop_words_list.extend(fp.readlines())

    cuted_result_frame = pd.read_csv(file_name, sep='\t', prefix='X', header=None)
    cuted_result_frame = cuted_result_frame.rename(columns={'X0': 'id', 'X1': 'hash_id', 'X2': 'words'})
    cuted_result_frame = cuted_result_frame.drop(['id'], axis=1)

    processed_corpus = cuted_result_frame['words'].map(lambda x: eval(x))
    dictionary = gensim.corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = gensim.models.TfidfModel(bow_corpus)

    some_doc = dictionary.doc2bow(processed_corpus.values[0])
    print(tfidf[some_doc])



def optimized_use_gensim_tfidf_package():
    import jieba.analyse
    import gensim

    cuted_result_file = 'use_jieba_extract_news_keywords.csv'

    write_to_csv_flag = False
    if write_to_csv_flag:
        stop_words_file = 'stop_words.txt'
        jieba.analyse.set_stop_words(os.path.join(os.getcwd(), stop_words_file))

        cuted_word_from_document = {}

        for file_item in os.listdir():
            if '.txt' in file_item and file_item != stop_words_file:
                news_hash_id = file_item.split('.')[0]
                content = None

                with open(file_item, 'r') as fp:
                    content = fp.read()

                tags = jieba.analyse.extract_tags(content, topK=10, withWeight=True)
                keywords = [tag_item[0] for tag_item in tags]
                cuted_word_from_document[news_hash_id] = {'hash_id': news_hash_id, 'words': tags, 'only_words': list(set(keywords))}

        cuted_result_frame = pd.DataFrame(cuted_word_from_document)
        cuted_result_frame = cuted_result_frame.T
        cuted_result_frame.to_csv(cuted_result_file, sep='\t', encoding='utf-8', header=False)
    
    cuted_result_frame = pd.read_csv(cuted_result_file, sep='\t', prefix='X', header=None)
    cuted_result_frame = cuted_result_frame.rename(columns={'X0': 'id', 'X1': 'hash_id', 'X2': 'only_words', 'X3': 'words'})
    cuted_result_frame = cuted_result_frame.drop(['id'], axis=1)

    processed_corpus = cuted_result_frame['only_words'].map(lambda x: eval(x))
    dictionary = gensim.corpora.Dictionary(processed_corpus.values)
    bow_corpus = []

    for item in cuted_result_frame.iterrows():
        bow_corpus_item = []
        tags = eval(item[1]['words'])

        for tag_item in tags:
            if tag_item[0] in dictionary.token2id.keys():
                bow_corpus_item.append((dictionary.token2id[tag_item[0]], tag_item[1]))

        bow_corpus.append(bow_corpus_item)

    tfidf = gensim.models.TfidfModel(bow_corpus)
    print(cuted_result_frame.iloc[0]['only_words'])
    print(cuted_result_frame.iloc[0]['words'])
    print(tfidf[dictionary.doc2bow(eval(cuted_result_frame.iloc[0]['only_words']))])
    

    



if __name__ == '__main__':
    url = 'http://news.sogou.com/'
    # request_sogou_news(url)


    file_name = 'list_news_url_0105.csv'
    # read_news_url_list(file_name)
    # read_correct_news_url_list(file_name)


    # cut_all_news_document()


    cuted_result_file = 'frame_cuted_news_content_words.csv'
    # use_gensim_package(cuted_result_file)

    corpus_file = 'corpus.bin'
    # load_corpus_and_process(corpus_file)



    # content_list = read_news_file()
    file_name = 'frame_cuted_news_content_words_train.csv'
    test_doc_content = []
    # train_corpus = list(read_news_file(file_name, test_doc_content))
    # use_gensim_doc2vec_package(train_corpus, test_doc_content)

    # use_gensim_tfidf_package(file_name)


    optimized_use_gensim_tfidf_package()
