conf = {
    'train': {
        'input': '/mnt/train.7z',
        'output_dir': 'data',
        'calc_stats': True,
        'asm_folder': 'train_asm',
        'labels': '../trainLabels.csv',
        'limit': None
        # 'limit': 100
    },
    'test': {
        'input': '/mnt/test.7z',
        'output_dir': 'data',
        'calc_stats': False,
        'asm_folder': 'test_asm/',
        'labels': '../trainLabels.csv',
        'limit': None
        # 'limit': 100
    },
    'sample': {
        'input': 'dataSample.7z',
        'output_dir': 'sample_data',
        'calc_stats': True,
        'asm_folder': 'sample_asm',
        'labels': '../trainLabels.csv'
    }
}

def out_folder(env):
    return conf[env]['output_dir']

def locate_file(env, fname):
    return '%s/%s' % (out_folder(env), fname)
