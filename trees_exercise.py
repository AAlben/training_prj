from math import log
import operator

import pandas as pd


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


class trees_exercise_iteration(object):
    def __init__(self):
        pass

    def calculate_shannon_entropy(self, data_frame):
        data_frame_shape = data_frame.shape
        shannon_entropy = 0

        if data_frame_shape[1] == 1:
            return shannon_entropy
        
        groupby_result = data_frame.groupby(data_frame.columns[-1]).count()
        for index in groupby_result.index:
            prob = float(groupby_result.loc[index].values[0]) / data_frame_shape[0]
            shannon_entropy -= prob * log(prob, 2)

        return shannon_entropy


    def split_data_set(self, data_frame, feature, feature_value):
        remained_data_frame = data_frame[data_frame[feature]==feature_value]
        remained_data_frame = remained_data_frame.drop(feature, axis=1)
        return remained_data_frame        


    def choose_best_feature_split(self, data_frame):
        base_shannon_entropy = self.calculate_shannon_entropy(data_frame)
        row_len = data_frame.shape[0]
                
        bestInfoGain = 0.0
        bestFeature = None

        for feature in data_frame.columns:
            if feature == data_frame.columns[-1]:
                continue

            new_shannon_entropy = 0.0
            for unique_feature_value in data_frame[feature].unique():             
                remained_data_frame = self.split_data_set(data_frame, feature, unique_feature_value)
                feature_shannon_entropy = self.calculate_shannon_entropy(remained_data_frame)
                prob = remained_data_frame.shape[0] / float(row_len)
                new_shannon_entropy += prob * feature_shannon_entropy

            sub_shannon_entropy = base_shannon_entropy - new_shannon_entropy
            if (sub_shannon_entropy > bestInfoGain):
                bestInfoGain = sub_shannon_entropy
                bestFeature = feature

        return bestFeature


    def create_tree(self, data_frame):
        data_frame_shape = data_frame.shape
        label_frame = data_frame[data_frame.columns[-1]]
        groupby_label_frame = label_frame.groupby(label_frame.values).count().sort_values(ascending=False)
        '''
            no     3
            yes    2
            Name: 2, dtype: int64
        '''

        if data_frame_shape[0] == groupby_label_frame[0]:
            return groupby_label_frame.index[0]

        if data_frame_shape[1] == 1:
            return groupby_label_frame.index[0]

        bestFeature = self.choose_best_feature_split(data_frame)
        my_tree = {
                    bestFeature: {}
        }

        for best_fetaure_value in data_frame[bestFeature].unique():
            remained_data_frame = self.split_data_set(data_frame, bestFeature, best_fetaure_value)

            '''
            remained_data_frame = 
                   1    2
                0  1  yes
                1  1  yes
                2  0   no
            or
                   1   2
                3  1  no
                4  1  no            
            '''

            my_tree[bestFeature][best_fetaure_value] = self.create_tree(remained_data_frame)


        return my_tree
        

class use_trees_classify(object):
    def __init__(self):
        pass

    def classify(self, input_tree, test_vec):
        first_node = list(input_tree.keys())[0]
        # first_node = 0

        second_dict = input_tree[first_node]
        '''
        second_dict = {
            0: 'no',
            1: {
                1列: {
                    0: 'no',
                    1: 'yes'
                }
            }
        }
        '''

        vec_value = test_vec[first_node]
        vec_value_result = second_dict[vec_value]
        if isinstance(vec_value_result, dict):
            class_label = self.classify(vec_value_result, test_vec)
        else:
            class_label = vec_value_result
        return class_label


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    '''
    dataSet = 
        [
            [1, 1, 'yes'], 
            [1, 1, 'yes'], 
            [1, 0, 'no'], 
            [0, 1, 'no'], 
            [0, 1, 'no']
        ]
    '''

    # classList = ['yes', 'yes', 'no', 'no', 'no']

    # if classList.count('yes') == len(classList):

    # if len(dataSet[0]) == 1:
    # len([1, 1, 'yes']) == 1

    data_frame = pd.DataFrame(dataSet)

    trees_exercise_obj = trees_exercise_iteration()
    my_tree = trees_exercise_obj.create_tree(data_frame)
    print(my_tree)
    
    '''
    my_tree = 
    {
        0列: {
            0: 'no',
            1: {
                1列: {
                    0: 'no',
                    1: 'yes'
                }
            }
        }
    }
    '''

    use_trees_classify_obj = use_trees_classify()
    test_vec = [1, 0]
    test_vec = [0]
    test_vec = [1, 1]
    # test_vec = [1]
    classify_result = use_trees_classify_obj.classify(my_tree, test_vec)
    print(classify_result)


