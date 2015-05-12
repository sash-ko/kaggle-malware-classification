from libarchive.public import file_reader
from multiprocessing import Pool, JoinableQueue, Queue

# Thanks to https://www.kaggle.com/users/159959/dmitry-ulyanov
# for the 7z archive traverser:
# https://www.kaggle.com/c/malware-classification/forums/t/12644/code-for-parallel-feature-extraction-without-unzipping-the-data

def crunch(file_name, ext_type, handler, pool_size=4, queue_size=40,
           limit=None):

    print 'Crunching file: %s, limit: %s' % (file_name, limit)

    q = JoinableQueue(queue_size)
    q_feats = Queue()

    pool = Pool(pool_size, wrap_handler(handler), ((q, q_feats),))

    with file_reader(file_name) as reader:
        idx = 0
        for entry in reader:

            if (entry.pathname.find(ext_type) != -1):
                text = [b for b in entry.get_blocks()]
                key = entry.pathname.split('/')[-1].split('.')[0]

                q.put((key, text), True)
                idx += 1

                print 'Processing:', entry.pathname, idx

                if limit and idx >= limit:
                    print 'Reached the limit'
                    break

        q.close()
        q.join()
        pool.close()

    result = []
    for i in range(q_feats.qsize()):
        result.append(q_feats.get())
    return result

def wrap_handler(fn):
    def wrapper((q_in, q_out)):
        while True:
            (key, text) = q_in.get()

            lines = ''.join(text).split('\r\n')

            result = fn(key, lines)

            q_out.put(result)
            q_in.task_done()
    return wrapper
