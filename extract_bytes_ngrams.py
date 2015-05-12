import pandas as pd
import ujson
from pe_parser import build_ngrams
from sevenz_cruncher import crunch
import config
import sys

ONE_GR_FILE = 'bytes_ngrams_1_{env}.pd'
TWO_GR_FILE = 'bytes_ngrams_2_{env}.txt'

def one_grams(key, lines):
    bytes_ = {}
    for line in lines:
        items = line.split()[1:]
        for item in items:
            if item == '??':
                item = -1
            else:
                item = int(item, 16)
            if item in bytes_:
                bytes_[item] = bytes_[item] + 1
            else:
                bytes_[item] = 1

    bytes_['key'] = key
    return bytes_

def two_grams(key, lines):
    bytes_ = []
    for line in lines:
        items = line.split()[1:]
        for item in items:
            if item == '??':
                bytes_.append(-1)
            else:
                bytes_.append(int(item, 16))

    ngrams = dict(build_ngrams([bytes_], 2))
    ngrams['key'] = key
    return ngrams.items()

def bytes_1_grams(input_, output, limit):
    print 'Processing', input_

    rows = []
    for row in crunch(input_, '.bytes', one_grams, limit=limit):
        rows.append(row)

    pd.DataFrame(rows).to_csv(output)
    print 'Saved to', output

def bytes_2_grams(input_, output, limit):
    print 'Processing', input_

    with open(output, 'w') as f:
        for row in crunch(input_, '.bytes', two_grams, limit=limit):
            f.write(ujson.dumps(row))
            f.write('\n')

    print 'Saved to', output

if __name__ == '__main__':
    env = sys.argv[1].lower()

    limit = config.conf[env].get('limit', None)

    one_gr_file = ONE_GR_FILE.format(env=env)
    bytes_1_grams(config.conf[env]['input'],
                  config.locate_file(env, one_gr_file), limit)

    two_gr_file = TWO_GR_FILE.format(env=env)
    bytes_2_grams(config.conf[env]['input'],
                  config.locate_file(env, two_gr_file), limit)
