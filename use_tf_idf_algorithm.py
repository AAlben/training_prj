
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



def load_data_file(file_path, file_name, encoding=None):
    file_name = os.path.join(file_path, file_name)
    data_frame = pd.read_csv(file_name, sep=';', encoding=encoding, error_bad_lines=False)
    return data_frame


if __name__ == '__main__':
    file_path = '/home/lichenguang/code/BookCrossing_Data'
    file_name_list = ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX-Books.csv']

    book_frame = load_data_file(file_path, file_name_list[2], 'ISO-8859-1')

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(book_frame['Book-Title'])

    print(tfidf_matrix)