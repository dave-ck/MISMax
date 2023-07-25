# import time
# import numpy as np
# from multiprocessing.pool import ThreadPool
from os import getpid
from multiprocessing import Pool
# from numba import njit, prange
# import networkx as nx


def hello(i):
    print("PID", getpid())
    return 2*i

if __name__ == "__main__":
    with Pool(processes=20) as pool:
        result = pool.map(hello, [i for i in range(10)])
        print(result)

