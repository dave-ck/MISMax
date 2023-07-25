# import time
# import numpy as np
# from multiprocessing.pool import ThreadPool
from os import getpid
from multiprocessing.pool import ThreadPool


# from numba import njit, prange
# import networkx as nx

2
def hello(i):
    print("PID", getpid())
    return 2 * i


if __name__ == "__main__":
    with ThreadPool(processes=20) as pool:
        result = pool.map(hello, [i for i in range(10)])
        print(result)
