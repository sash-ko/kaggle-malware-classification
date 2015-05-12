import numpy as np
import pe_parser
from sevenz_cruncher import crunch
import config
import sys

def process_asm(key, lines, folder):
    features = pe_parser.parse(lines)
    np.savez_compressed(folder % key, features)
    return 1

if __name__ == '__main__':
    env = sys.argv[1]

    folder = config.locate_file(env, config.conf[env]['asm_folder'])
    limit = config.conf[env].get('limit', None)
    crunch(config.conf[env]['input'], '.asm',
           lambda key, lines: process_asm(key, lines, folder + '/%s'),
           limit=limit)
