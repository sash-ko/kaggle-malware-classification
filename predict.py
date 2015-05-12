import numpy as np
import random
import pandas as pd
from ml_tools import cross_valid, predict, dump_submission, permuate

RANDOM_SEED = 10
OUTPUT = 'prediction.csv'

PARAMS = {
    'bst:max_depth': 3,
    'bst:eta': 0.1,
    'silent': 1,
    'colsample_bytree': 0.4,
    'objective': 'multi:softprob',
    'min_child_weight': 1,
    'subsample': 0.9,
    'num_class': 9,
    'nthread': 4,
    'seed': RANDOM_SEED
}
ITERATIONS = 300

def make_prediction(data_train, labels, data_test, indexes):
    predictions = predict(data_train, labels, data_test, PARAMS, ITERATIONS)
    dump_submission(predictions, indexes, OUTPUT)

def make_data_group(names, labels):
    first = load_features(names[0], need_sort=True, keep_key=True)

    keys = first['key'].values
    first.drop('key', axis=1, inplace=True)

    dsets = [first]
    for name in names[1:]:
        data = load_features(name, need_sort=True, keep_key=False)
        dsets.append(data.values)

    data = pd.DataFrame(np.hstack(dsets))
    if labels:
        data['target'] = labels

    data['key'] = keys
    data = permuate(data)

    keys = data['key'].values
    data.drop('key', axis=1, inplace=True)

    if labels:
        labels = data['target'].values
        data.drop('target', axis=1, inplace=True)
        return data, keys, labels

    return data, keys

def convert_files(files, env):
    return [f % env for f in files]

def load_features(name, need_sort=True, keep_key=False):
    print 'Loading file:', name
    data = pd.read_csv(name)
    data = data[data.columns[1:]]

    key_name = '0'

    data['key'] = data[key_name].values
    data.drop(key_name, axis=1, inplace=True)
    key_name = 'key'

    if need_sort:
        data.sort(key_name, inplace=True)

    if not keep_key:
        data.drop(key_name, axis=1, inplace=True)

    return data.fillna(0)

def load_labels():
    labels = pd.read_csv('trainLabels.csv')
    labels.sort('Id', inplace=True)
    return [c - 1 for c in labels.Class]

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    files = [
        'data/opcodes_6_model_sections_%s.pd',

        'data/bytes_ngrams_1_%s.pd',
        'data/bytes_ngrams_2_top_1000_%s.pd',

        'data/opcodes_1_model_%s.pd',

        'data/opcodes_2_model_2gram_top_1000_%s.pd',
        'data/opcodes_3_model_3gram_top_1000_%s.pd',
        'data/opcodes_4_model_4gram_top_1000_%s.pd',
        'data/opcodes_5_model_5gram_top_1000_%s.pd',

        'data/opcodes_7_model_misc_%s.pd',

        # 'opcodes_7_model_dlls_%s.pd',
    ]

    data_train, _, labels = make_data_group(convert_files(files, 'train'),
                                            load_labels())

    data_test, indexes = make_data_group(convert_files(files, 'test'),
                                         None)

    print 'data_train shape', data_train.shape

    idx = data_test.max(axis=0) > 0
    data_train = data_train[data_train.columns[idx]]
    data_test = data_test[data_test.columns[idx]]

    cross_valid(data_train, labels, PARAMS, ITERATIONS, 6, silent=False)

    make_prediction(data_train, labels, data_test, indexes)

