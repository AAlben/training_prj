


import pandas as pd
import numpy as np


def get_news_file():
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


def init_now_news(now_news_key):
    # read new news from mysql

    news_list = get_news_file()

    data_frame = pd.DataFrame(news_list, columns='content')

    print(data_frame)


    pass


if __name__ == '__main__':
    now_news_key = 'This is Now News Redis Key'
    init_now_news(now_news_key)