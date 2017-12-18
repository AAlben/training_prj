
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os



def load_data_file(file_path, file_name, encoding=None):
    file_name = os.path.join(file_path, file_name)
    data_frame = pd.read_csv(file_name, sep=';', encoding=encoding, error_bad_lines=False)
    return data_frame


if __name__ == '__main__':
    file_path = '/home/lichenguang/code/BookCrossing_Data'
    file_name_list = ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX-Books.csv']

    book_frame = load_data_file(file_path, file_name_list[2], 'ISO-8859-1')
    book_frame = book_frame.iloc[0:15000]

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(book_frame['Book-Title'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    print(tfidf_matrix[2])
    print(tfidf_matrix[2].shape)

    print(cosine_similarities)
    print(cosine_similarities.shape)

    recommend_result = {}

    for idx, row in book_frame.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], book_frame['Book-Title'][i]) for i in similar_indices]
        print(similar_items)

        flattened = sum(similar_items[1:], ())

        recommend_result[similar_items[0][1]] = flattened


    print(recommend_result)