from os import getpid
from multiprocessing import Pool, current_process
from numba import njit
import numpy as np


@njit
def hello(i):
    # print(f"PID: {getpid()}, Process name: {current_process().name}")
    return 2 * i


if __name__ == "__main__":
    with Pool() as pool:
        result = pool.map(hello, np.random.random((1024, 3, 3)))
        print(result)
