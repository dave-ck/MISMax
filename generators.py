import numpy as np
from numba import njit


# @njit # todo: figure out njit here
def filter_generator(source_iterable, flag_iterable):
    """
    Yields sources where corresponding flags are True.
    :param source_iterable: Values for filtering.
    :param flag_iterable: Iterable of booleans (or ints or whatever). Must be of equal length to source_iterable.
    :return: sources whenever the corresponding flag is True
    """
    for obj, flag in zip(source_iterable, flag_iterable):
        if flag:
            yield obj


@njit
def shifty_bitstring_generator(n):
    A = np.empty(n, dtype=np.uint8)
    for i in range(2 ** n):
        for j in range(n - 1, -1, -1):
            A[j] = i & 1
            i >>= 1
        # A contains the binary representation of i
        yield A
    return


@njit
def ehrlich_permutation_generator(n):
    a = np.arange(n, dtype=np.uint8)
    b = np.arange(n, dtype=np.uint8)
    c = np.zeros(n + 1, dtype=np.uint8)  # c[0] should never be accessed; using arr of length n+1 to index starting at 1
    while True:
        yield a.copy()
        k = 1
        while c[k] == k:
            c[k] = 0
            k += 1
        if k == n:
            return  # is break better?
        c[k] += 1
        a[0], a[b[k]] = a[b[k]], a[0]
        # reversing sublist as in textbook - swap for numpy impl. later if compatible with njit
        j = 1
        k = k - 1
        while j < k:
            b[j], b[k] = b[k], b[j]
            j += 1
            k -= 1
