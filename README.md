# FastAnnoy

This library is a [pybind11](https://github.com/pybind/pybind11) port of [spotify/annoy](https://github.com/spotify/annoy).

# Installation 

To install, just do `pip install fastannoy` to pull down from [PyPI](https://pypi.python.org/pypi/fastannoy).

## Install from source code

- clone this repository
- `pip install ./fastannoy`

# Backgroud

First of all, thanks for spotify/annoy's awesome work, it provides efficient implement for Approximate Nearest Neighbors Search. But when i find that batch search is missing, so this project's initial purpose is for batch search.

However, it's written in pybind11 for python interface, and discovered better performance.

# Usage

All basic interfaces is same as [spotify/annoy](https://github.com/spotify/annoy?tab=readme-ov-file#full-python-api).

```python
from fastannoy import AnnoyIndex
import random

f = 40  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')
for i in range(1000):
    v = [random.gauss(0, 1) for _ in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 100)) # will find the 100 nearest neighbors
"""
[0, 17, 389, 90, 363, 482, ...]
"""

print(u.get_nns_by_vector([random.gauss(0, 1) for _ in range(f)], 100)) # will find the 100 nearest neighbors by vector
"""
[378, 664, 296, 409, 14, 618]
"""
```

## Batch Search

Corresponding to `get_nns_by_item`, the batch search version is `get_batch_nns_by_items`. The first argument should be a list of int.

In the same way, corresponding to `get_nns_by_vector`, the batch search version is `get_batch_nns_by_vectors`. The first argument should be a list of list[int].

And the batch search's implement supports multiple threads. You can set the argument `n_threads`, the default is 1.

```python
# will find the 100 nearest neighbors

print(u.get_batch_nns_by_items([0, 1, 2], 100))
"""
[[0, 146, 858, 64, 833, 350, 70, ...], 
[1, 205, 48, 396, 382, 149, 305, 125, ...], 
[2, 898, 503, 618, 23, 959, 244, 10, 445, ...]]
"""

print(u.get_batch_nns_by_vectors([
    [random.gauss(0, 1) for _ in range(f)]
    for _ in range(3)
], 100))
"""
[[862, 604, 495, 638, 3, 246, 778, 486, ...], 
[260, 722, 215, 709, 49, 248, 539, 126, 8, ...], 
[288, 764, 965, 320, 631, 505, 350, 821, 540, ...]]
"""
```

# Benchmark

The results are running in my macbook with the [test script](https://github.com/QunBB/fastannoy/blob/main/examples/performance_test.py), so focus on time consumption relatively between fastannoy and annoy.

|                                                    | fastannoy      | annoy          |
| -------------------------------------------------- | -------------- | -------------- |
| **50W items with 128 dimension**                   |                |                |
| - build+add_item                                   | 13.810 seconds | 19.633 seconds |
| - 5W times search                                  | 20.613 seconds | 39.760 seconds |
| - 5k times search with 10 batch size and 5 threads | 6.542 seconds  | /              |



