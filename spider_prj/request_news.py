import requests


def request_sogou_news(url):
    r = requests.get(url)
    print(r.text)



if __name__ == '__main__':
    url = 'http://news.sogou.com/'

    request_sogou_news(url)