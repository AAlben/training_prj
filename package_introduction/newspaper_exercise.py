from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

import os
import re

import jieba.analyse



def parse_url_html(url, file_name):
    # url = 'http://www.sohu.com/a/211570823_115565?_f=index_chan30news_1'
    article = Article(url, language='zh')
    article.download()

    article.parse()
    news_content = article.text
    # news_content = article.html

    # file_name = 'news_html.txt'
    with open(file_name, 'w') as fp:
        fp.write(news_content)


def parse_original_html(file_name):
    page_html = None
    with open(file_name, 'r') as fp:
        page_html = fp.readlines()

    page_html = ''.join(page_html)
    html = lxml.html.fromstring(page_html)
    
    html_body = html.body
    pass




if __name__ == '__main__':

    use_tf_idf_flag = False
    count_vectorizer_flag = False
    crawl_souhu_flag = False
    use_jieba_flag = True

    news_list = []
    project_path = os.getcwd()
    for file_item in os.listdir(project_path):
        if '.txt' in file_item:            
            file_name = file_item

            if 'news_html.txt' == file_name:
                continue

            news_content = None
            with open(file_name, 'r') as fp:
                news_content = fp.readlines()
            news_content = ''.join(news_content)

            news_list.append(news_content)

    if use_tf_idf_flag:        
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, analyzer='word', ngram_range=(1, 3))
        # tfidf = transformer.fit_transform(tfidf_vectorizer.fit_transform(news_list))
        tfidf_matrix = tfidf_vectorizer.fit_transform(news_list)
        word = tfidf_vectorizer.get_feature_names()
        weight = tfidf_matrix.toarray()
        # print(weight.shape)
        # weight.shape = (7, 428)
        # tf.idf_.shape = (5577,)
        # tfidf_matrix.shape = (7, 5577)

        similar_matrix = tfidf_matrix * tfidf_matrix.T
        print(similar_matrix)

    elif count_vectorizer_flag:
        transformer = TfidfTransformer(smooth_idf=True)
        counts = [[3, 0, 1],
                   [2, 0, 0],
                   [3, 0, 0],
                   [4, 0, 0],
                   [3, 2, 0],
                   [3, 0, 2]]
        tfidf = transformer.fit_transform(counts)
        print(transformer.idf_)
        '''
        print(tfidf.toarray())
        [[ 0.81940995  0.          0.57320793]
         [ 1.          0.          0.        ]
         [ 1.          0.          0.        ]
         [ 1.          0.          0.        ]
         [ 0.47330339  0.88089948  0.        ]
         [ 0.58149261  0.          0.81355169]]
        '''
    
    elif crawl_souhu_flag:
        url_list = ['http://www.sohu.com/a/211828117_413981',
                    'http://www.sohu.com/a/211809255_115565',
                    'http://www.sohu.com/a/211799939_118680',
                    'http://www.sohu.com/a/211801914_161623',
                    'http://www.sohu.com/a/211819311_114837']


        for url in url_list:
            parse_url_html(url, '{0}.txt'.format(url.split('/')[-1]))

    elif use_jieba_flag:
        term_set = []

        for news in news_list:
            extract_result = jieba.analyse.extract_tags(news, topK = 50)
            term_set.extend(extract_result)

        term_set = set(term_set)

        count_matrix = []
        for term in term_set:            
            pass
