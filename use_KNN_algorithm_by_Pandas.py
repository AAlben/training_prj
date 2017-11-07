import pandas as pd
import numpy as np
import operator


def load_data_file(file_name):
    data_frame = pd.read_csv(file_name, sep=',', header=None)
    return data_frame


def use_KNN_algorithm(basic_data, test_data):
    collect_data, label_data = collectiong_data(basic_data)
    frame_size = collect_data.shape

    artument_K = 10

    test_collect_data, test_label_data = collectiong_data(test_data)
    for test_item in test_collect_data.values:
        diff_data = collect_data - test_item
        square_diff_data = diff_data ** 2
        sum_square_diff_data = square_diff_data.sum(axis=1)
        calculate_result_data = sum_square_diff_data ** 0.5
        show_sorted_index_result = calculate_result_data.argsort()
        
        test_item_label_count = {}

        for range_index in range(artument_K):
            range_index_label = label_data[show_sorted_index_result[range_index]]
            test_item_label_count[range_index_label] = test_item_label_count.get(range_index_label, 0) + 1
        
        sorted_test_item_label = sorted(test_item_label_count.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_test_item_label)


def collectiong_data(basic_data):
    num_value_frame = basic_data.ix[:, 0:3]
    label_frame = basic_data.ix[:, 4]

    min_val = num_value_frame.min()
    max_val = num_value_frame.max()
    frame_size = num_value_frame.shape
    empty_arrays = np.zeros(frame_size)
    min_data_frame = pd.DataFrame(empty_arrays)

    min_data_frame.loc[:, :] = min_val.values
    molecular = num_value_frame - min_data_frame
    denominator = pd.DataFrame(empty_arrays)
    denominator.loc[:, :] = (max_val - min_val).values

    collect_data = molecular / denominator
    return collect_data, label_frame


if __name__ == '__main__':
    file_name_list = ['bezdekIris.data', 'iris.data']
    bezdekIris_frame = load_data_file(file_name_list[0])
    iris_frame = load_data_file(file_name_list[1])

    use_KNN_algorithm(bezdekIris_frame, iris_frame)

