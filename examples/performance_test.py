import time
from fastannoy import AnnoyIndex
from annoy import AnnoyIndex as BasicAnnoyIndex
import numpy as np

f = 128
total = 500000
times = 50000


def run(annoy_cls):
    st = time.time()

    ann = annoy_cls(f, 'euclidean')

    for i in range(total):
        ann.add_item(i, np.random.random(f))

    ann.build(50, -1)

    print(f'build time: {time.time() - st}')

    st = time.time()
    for i in range(times):
        ann.get_nns_by_item(i, 200, 2000, False)
    print(f'search time: {time.time() - st}')


def batch_run(bs=10):

    ann = AnnoyIndex(f, 'euclidean')

    for i in range(total):
        ann.add_item(i, np.random.random(f))

    ann.build(50, -1)

    st = time.time()
    for i in range(int(times / bs)):
        ann.get_batch_nns_by_items(list(range(i * bs, (i + 1) * bs)), 200, 2000, False, n_threads=5)
    print(f'batch search time: {time.time() - st}')


if __name__ == '__main__':
    run(AnnoyIndex)
    run(BasicAnnoyIndex)
    batch_run()
