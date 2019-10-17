import math
import numpy as np
from itertools import permutations

array = np.zeros((6, 6), dtype=np.int32)

def check(array):
    for kx in range(0, 3):
        for ky in range(0, 3):
            slice = array[kx:kx+4, ky:ky+4].reshape(-1)
            diff = np.diff(slice)
            if len(np.unique(diff)) != 1:
                return False
        return True

indices = tuple(np.arange(0, 36).tolist())
total = math.factorial(36)
ind = 0
for perm in permutations(indices):
    if ind % (1024 * 1024 * 1024) == 0:
        print(f"\r Finished {ind}-{total} {ind/total}", flush=True)
    ind += 1
    perm = np.asarray(perm).reshape(6, 6)
    if check(perm):
        print("Found!")
        print(perm)
        break
