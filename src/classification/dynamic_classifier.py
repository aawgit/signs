#https://github.com/slaypni/fastdtw

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_test():
    x = np.array([[1, 1, 1], [2, 2, 4], [3, 3, 7], [4, 4, 1], [1, 1, 1]])
    y = np.array([[2, 2, 4], [3, 3, 7], [4, 4, 1]])
    distance, path = fastdtw(x, y, dist=euclidean)
    print(distance, path)