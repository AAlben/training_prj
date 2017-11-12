import numpy as np

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


def load_data_file(file_name):
    dataset = np.loadtxt(file_name, delimiter=',')
    dataset_shape = dataset.shape
    x = dataset[:, 0:dataset_shape[1] - 2]
    y = dataset[:, dataset_shape[1] - 1]

    return x, y


if __name__ == '__main__':
    file_name = 'pima-indians-diabetes.data'
    x, y = load_data_file(file_name)

    model = DecisionTreeClassifier()
    model.fit(x, y)

    '''
    dir(model) = 
    ['__abstractmethods__', 
    '__class__', 
    '__delattr__', 
    '__dict__', 
    '__dir__', 
    '__doc__', 
    '__eq__', 
    '__format__', 
    '__ge__', 
    '__getattribute__', 
    '__getstate__', 
    '__gt__', 
    '__hash__', 
    '__init__', 
    '__le__', 
    '__lt__', 
    '__module__', 
    '__ne__', 
    '__new__', 
    '__reduce__', 
    '__reduce_ex__', 
    '__repr__', 
    '__setattr__', 
    '__setstate__', 
    '__sizeof__', 
    '__str__', 
    '__subclasshook__', 
    '__weakref__', 
    '_abc_cache', 
    '_abc_negative_cache', 
    '_abc_negative_cache_version', 
    '_abc_registry', 
    '_estimator_type', 
    '_get_param_names', 
    '_validate_X_predict', 
    'apply', 
    'class_weight', 
    'classes_', 
    'criterion', 
    'decision_path', 
    'feature_importances_', 
    'fit', 
    'get_params', 
    'max_depth', 
    'max_features', 
    'max_features_', 
    'max_leaf_nodes', 
    'min_impurity_decrease', 
    'min_impurity_split', 
    'min_samples_leaf', 
    'min_samples_split', 
    'min_weight_fraction_leaf', 
    'n_classes_', 
    'n_features_', 
    'n_outputs_', 
    'predict', 
    'predict_log_proba', 
    'predict_proba', 
    'presort', 
    'random_state', 
    'score', 
    'set_params', 
    'splitter', 
    'tree_']
    '''
