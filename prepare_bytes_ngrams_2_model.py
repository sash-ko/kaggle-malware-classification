import ujson
from collections import Counter
import pandas as pd
import config
import sys

MAX_NGRAMS_FILE = 'max_bytes_ngrams_top_%s.txt'
DATA_FILE = 'bytes_ngrams_2_%s.txt'
OUTPUT_FILE = 'bytes_ngrams_2_top_%s_%s.pd'

def key_to_str(key):
    if isinstance(key, list):
        key = "%s_%s" % tuple(key)
    return key

def calc_top_n_ngrams(env, n):
    counter = Counter()

    fname = config.locate_file(env, DATA_FILE % env)
    with open(fname) as f:
        for line in f:
            record = ujson.loads(line.strip())
            for k, value in record:
                if k == 'key':
                    continue

                k = key_to_str(k)
                counter[k] = counter[k] + 1
    counter = counter.items()
    counter.sort(key=lambda v: v[1], reverse=True)
    counter = dict(counter[:n])

    fname = config.locate_file(env, MAX_NGRAMS_FILE % n)
    with open(fname, 'w') as f:
        f.write(ujson.dumps(counter))
        print 'Saved to', fname

def load_max_keys(env, n):
    fname = config.locate_file(env, MAX_NGRAMS_FILE % n)
    with open(fname) as f:
        return ujson.loads(''.join(f.readlines()))

def build_model(env, n):
    max_keys = load_max_keys(env, n)
    max_keys = max_keys.keys()
    max_keys.sort()

    features = []
    fname = config.locate_file(env, DATA_FILE % env)
    with open(fname) as f:
        for line in f:
            record = ujson.loads(line.strip())
            record = dict([(key_to_str(k), v) for k, v in record])

            current = [record['key']] + [record.get(k, 0) for k in max_keys]
            features.append(current)

    features = pd.DataFrame(features)
    fname = config.locate_file(env, OUTPUT_FILE % (n, env))
    features.to_csv(fname)
    print 'Saved to', fname


if __name__ == '__main__':
    env = sys.argv[1]

    n = 1000
    if config.conf[env]['calc_stats']:
        calc_top_n_ngrams(env, n)

    build_model(env, n)
