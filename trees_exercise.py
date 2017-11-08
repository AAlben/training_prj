from math import log
import operator

import pandas as pd



class trees_exercise(object):
    def __init__(self, dataSet):
        self.dataSet = dataSet

    def calculate_shannon_entropy(self, data_frame):        
        row_len = data_frame.shape[0]
        shannon_entropy = 0

        groupby_result = data_frame.groupby(data_frame.columns[-1]).count()
        for index in groupby_result.index:
            prob = float(groupby_result.loc[index].values[0]) / row_len
            shannon_entropy -= prob * log(prob, 2)

        return shannon_entropy, data_frame
        

    def choose_best_feature_split(self, dataSet):
        base_shannon_entropy, data_frame = self.calculate_shannon_entropy(dataSet)
        value_frame = data_frame
        row_len = value_frame.shape[0]
                
        bestInfoGain = 0.0
        bestFeature = None

        for feature in value_frame.columns:
            if feature == value_frame.columns[-1]:
                continue

            new_shannon_entropy = 0.0
            for unique_feature_value in value_frame[feature].unique():             
                remained_value_frame = self.split_data_set(value_frame, feature, unique_feature_value)
                feature_shannon_entropy, remained_value_frame = self.calculate_shannon_entropy(remained_value_frame)

                print('***remained***')
                print(remained_value_frame)
                print('***remained***')

                prob = remained_value_frame.shape[0] / float(row_len)
                new_shannon_entropy += prob * feature_shannon_entropy

            sub_shannon_entropy = base_shannon_entropy - new_shannon_entropy
            if (sub_shannon_entropy > bestInfoGain):
                bestInfoGain = sub_shannon_entropy
                bestFeature = feature

        return bestFeature


    def split_data_set(self, value_frame, feature, feature_value):
        remained_value_frame = value_frame[value_frame[feature]==feature_value]
        remained_value_frame = remained_value_frame.drop(feature, axis=1)
        return remained_value_frame


    def create_tree(self, data_frame, label_frame):        
        row_len = data_frame.shape[0]
        groupby_label_frame = label_frame.groupby(label_frame.values).count()
        label_len = label_frame.shape[0]

        if row_len == groupby_label_frame.shape[0]:
            return label_frame.iloc[0]

        if label_len == 1:
            return None

        bestFeature = self.choose_best_feature_split(data_frame)
        my_tree = {
                    bestFeature: {}
        }

        for best_fetaure_value in data_frame[bestFeature].unique():
            remained_value_frame = self.split_data_set(data_frame, bestFeature, best_fetaure_value)
            my_tree[bestFeature][best_fetaure_value] = self.create_tree(remained_value_frame, label_frame)

        print(my_tree)


if __name__ == '__main__':
    dataSet, labels = createDataSet()

    trees_exercise_obj = trees_exercise(dataSet)

    data_frame = pd.DataFrame(dataSet)

    shannon_entropy, data_frame = trees_exercise_obj.calculate_shannon_entropy(data_frame)
    label_frame = data_frame[data_frame.columns[-1]]
    value_frame = data_frame.iloc[:, 0:len(data_frame.columns) - 1]
    trees_exercise_obj.create_tree(value_frame, label_frame)
    # bestFeature = trees_exercise_obj.choose_best_feature_split(dataSet)