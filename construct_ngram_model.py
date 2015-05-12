import os
import numpy as np
import pandas as pd
import ujson
import cPickle as pickle
import config
import sys


NUM_CLASSES = 9

def get_all_keys(folder, env):
    total_features = 7
    sets = {}
    for i in range(total_features):
        sets[i] = set()

    counter = 0
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            key = file_name.split('/')[-1]
            key = key.split('.')[0]
            file_name = '%s/%s' % (root, file_name)

            data = np.load(file_name)

            for fn in range(total_features):
                feature_set = data.items()[0][1][fn]
                sets[fn].update(list(feature_set[:, 0]))

            counter += 1

            if counter % 100 == 0:
                all_featues = sets.items()
                all_featues.sort(key=lambda x: x[0])
                print counter, [len(v) for k, v in all_featues]

    all_featues = sets.items()
    all_featues.sort(key=lambda x: x[0])
    print 'Results:', [len(v) for k, v in all_featues]

    sets = [to_array(v) for k, v in all_featues]

    fname = config.locate_file(env, 'all_featuresets_keys')
    np.savez_compressed(fname, sets)
    print 'saved to', fname

def count_n_fset_stats(env, n, folder):
    print 'Counting featureset stats', n

    labels = pd.read_csv(config.conf[env]['labels'])
    labels.set_index('Id', inplace=True)

    sum_per_class = {}
    count_per_class = {}

    counter = 0
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            key = file_name.split('/')[-1]
            key = key.split('.')[0]

            cls = int(labels.ix[key].Class) - 1

            file_name = '%s/%s' % (root, file_name)

            data = np.load(file_name)

            fset = dict(data.items()[0][1][n - 1])

            sum_d = sum_per_class.get(cls, {})
            count_d = count_per_class.get(cls, {})
            for k, v in fset.iteritems():
                if k in sum_d:
                    sum_d[k] = sum_d[k] + v
                    count_d[k] = count_d[k] + 1
                else:
                    sum_d[k] = v
                    count_d[k] = 1
            sum_per_class[cls] = sum_d
            count_per_class[cls] = count_d

            counter += 1

            if counter % 100 == 0:
                print 'processed', counter

    total = {'sum': sum_per_class, 'count': count_per_class}

    fname = config.locate_file(env, 'featureset_stats_%s.pkl' % n)
    with open(fname, 'w') as f:
        pickle.dump(total, f)

def to_array(st):
    return np.array(list(set(list(st))))

def constuct_n_model(env, n, folder, all_sets, force_str, postfix,
                     known_vals=None):
    print 'Constructing featureset', n

    all_keys = list(all_sets[n - 1])
    if isinstance(all_keys[0], np.ndarray):
        all_keys = [tuple(k) for k in all_keys]
    all_keys.sort()

    model = construct_n_model_using_keys(env, n, folder, all_sets, all_keys,
                                         force_str, known_vals)

    model = pd.DataFrame(model)
    fname = config.locate_file(env, 'opcodes_%s_model_%s.pd' % (n, postfix))
    model.to_csv(fname)
    print 'saved to', fname

def top_by_freq(env, n, top_n):
    keys = []
    fname = config.locate_file(env, 'featureset_stats_%s.pkl' % n)
    with open(fname) as f:
        stats = pickle.load(f)

    freqs = compute_freq(stats)

    for i in range(NUM_CLASSES):
        if i in freqs:
            class_freqs = list(freqs[i])
            class_freqs.sort(key=lambda x: x[1], reverse=True)
            keys.extend(class_freqs[: top_n])

    keys = list(set(keys))
    keys.sort()
    return keys

def top_unique(env, n, top_n):
    # keys = []
    fname = config.locate_file(env, 'featureset_stats_%s.pkl' % n)
    with open(fname) as f:
        stats = pickle.load(f)

    all_keys = {}
    for i in range(NUM_CLASSES):
        for k in stats['count'][i].iterkeys():
            if k not in all_keys:
                all_keys[k] = 1
            else:
                all_keys[k] = all_keys[k] + 1

    unique = set([k for k, v in all_keys.iteritems() if v == 1])

    columns = []
    for i in range(NUM_CLASSES):
        counts = [(k, v) for k, v in stats['count'][i].iteritems() if k in unique]
        counts.sort(key=lambda x: x[1], reverse=True)

        columns.extend([k for k, v in counts[:top_n]])

    return columns

def constuct_n_model_top_n(env, n, folder, all_sets, top_n_fn, postfix):
    print 'Constructing top n model', n, postfix

    all_keys = top_n_fn(n)

    print 'Max key number', len(all_keys)

    model = construct_n_model_using_keys(env, n, folder, all_sets, all_keys)

    model = pd.DataFrame(model)
    fname = config.locate_file(env, 'opcodes_%s_model_%s.pd' % (n, postfix))
    model.to_csv(fname)
    print 'saved to file', fname

def compute_freq(stats):
    sums = stats['sum']
    counts = stats['count']

    freqs = {}
    for i in range(NUM_CLASSES):
        current_freq = {}
        for i in sums:
            sum_vals = sums[i]
            count_vals = counts[i]

            for key, val in sum_vals.iteritems():
                cval = count_vals[key]
                current_freq[key] = val / float(cval)
            freqs[i] = current_freq

    return freqs

def construct_n_model_using_keys(env, n, folder, all_sets, all_keys,
                                 force_str=False, known_vals=None):
    model = []
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            key = file_name.split('/')[-1]
            key = key.split('.')[0]
            file_name = '%s/%s' % (root, file_name)

            data = np.load(file_name)

            fsets = data.items()[0][1][n - 1]
            if fsets.shape == ():
                fsets = np.array(fsets.item(0).items())

            fsets = dict(fsets)
            if force_str:
                fsets = {str(k): v for k, v in fsets.iteritems()}

            # NOTE: this is new and not tested stuff
            std = np.std([int(v) for v in fsets.itervalues()])
            total_count = len(fsets)
            total_sum = sum([int(v) for v in fsets.itervalues()])

            known_counter = []
            unknown_counter = []
            if known_vals:
                kv = [k for k in fsets.iterkeys() if k in known_vals]
                known_counter = [len(kv)]

                ukv = [k for k in fsets.iterkeys() if k not in known_vals]
                unknown_counter = [len(ukv)]

            fsets = np.array([key] +
                             [int(fsets.get(k, 0)) for k in all_keys] +
                             [std, total_count, total_sum] + known_counter +
                             unknown_counter,
                             dtype=np.object)
            model.append(fsets)

            if len(model) % 100 == 0:
                print 'Processed', len(model)
    return model

# def extract_known_keys(all_sets):
#     print 'Extracting known keys'
#     keys = {k: 1 for k in all_sets[0]}

#     save_json('train_set_mappings.txt', keys)

# def extract_feature_mappings(all_sets):
#     groups = [
#         ('sections', 6, None),
#         ('ngrams_1', 1, None),
#         ('ngrams_2', 2, lambda n: top_by_freq(n, 1000)),
#         ('ngrams_3', 2, lambda n: top_by_freq(n, 1000)),
#         ('ngrams_4', 2, lambda n: top_by_freq(n, 1000)),
#         ('ngrams_5', 2, lambda n: top_by_freq(n, 1000))
#     ]

#     mappings = {}
#     for name, n, fn in groups:
#         if fn:
#             keys = fn(n)
#         else:
#             keys = list(all_sets[n - 1])

#         mappings[name] = [(k, i) for i, k in enumerate(keys)]

#     save_json('all_mappings.txt', mappings)

def save_json(fname, object_):
    with open(fname, 'w') as f:
        f.write(ujson.dumps(object_))
    print 'Saved to', fname

if __name__ == '__main__':
    env = sys.argv[1]
    folder = config.locate_file(env, config.conf[env]['asm_folder'])

    if config.conf[env]['calc_stats']:
        get_all_keys(folder, env)

    fname = config.locate_file(env, 'all_featuresets_keys.npz')
    all_sets = np.load(fname).items()[0][1]

    constuct_n_model(env, 1, folder, all_sets, False, env)

    if config.conf[env]['calc_stats']:
        count_n_fset_stats(env, 2, folder)
        count_n_fset_stats(env, 3, folder)
        count_n_fset_stats(env, 4, folder)
        count_n_fset_stats(env, 5, folder)

    constuct_n_model_top_n(env, 2, folder, all_sets,
                           lambda n: top_by_freq(env, n, 1000),
                           '2gram_top_1000_%s' % env)
    constuct_n_model_top_n(env, 3, folder, all_sets,
                           lambda n: top_by_freq(env, n, 1000),
                           '3gram_top_1000_%s' % env)
    constuct_n_model_top_n(env, 4, folder, all_sets,
                           lambda n: top_by_freq(env, n, 1000),
                           '4gram_top_1000_%s' % env)
    constuct_n_model_top_n(env, 5, folder, all_sets,
                           lambda n: top_by_freq(env, n, 1000),
                           '5gram_top_1000_%s' % env)

    constuct_n_model(env, 6, folder, all_sets, False, 'sections_%s' % env)
    constuct_n_model(env, 7, folder, all_sets, False, 'misc_%s' % env)
