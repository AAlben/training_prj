import pandas as pd
import numpy as np
import operator


def load_data_file(file_name):
    data_frame = pd.read_csv(file_name, sep=',', header=None)
    return data_frame


def use_KNN_algorithm(basic_data, test_data):
    collect_data, label_data = collectiong_data(basic_data)
    row_len = collect_data.shape[0]

    artument_K = 10

    test_data_narray = test_data.as_matrix(columns=basic_data.columns[1:-1])
    for test_item in test_data_narray:
        diff_data = np.tile(test_item, (row_len, 1)) - collect_data
        square_diff_data = diff_data ** 2
        sum_square_diff_data = square_diff_data.sum(axis=1)
        calculate_result_data = sum_square_diff_data ** 0.5
        show_sorted_index_result = calculate_result_data.argsort()

        test_item_label_count = {}

        for range_index in range(artument_K):
            range_index_label = label_data[show_sorted_index_result[range_index]]
            test_item_label_count[range_index_label] = test_item_label_count.get(range_index_label, 0) + 1
        
        sorted_test_item_label = sorted(test_item_label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        print(sorted_test_item_label)



def collectiong_data(basic_data):
    basic_data_narray = basic_data.as_matrix(columns=basic_data.columns[1:-1])

    min_val = basic_data_narray.min(0)
    max_val = basic_data_narray.max(0)
    row_len = basic_data_narray.shape[0]

    molecular = basic_data_narray - np.tile(min_val, (row_len, 1))    
    denominator = np.tile(max_val - min_val, (row_len, 1))

    collect_data = molecular / denominator
    return collect_data, basic_data[4]


if __name__ == '__main__':
    file_name_list = ['bezdekIris.data', 'iris.data']
    bezdekIris_frame = load_data_file(file_name_list[0])
    iris_frame = load_data_file(file_name_list[1])

    # print(bezdekIris_frame)

    use_KNN_algorithm(bezdekIris_frame, iris_frame)

    pass