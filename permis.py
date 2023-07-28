from generators import shifty_bitstring_generator, ehrlich_permutation_generator
from numba import njit
import numpy as np


@njit
def is_permis(M, word):
    n = M.shape[0]
    for start_status in shifty_bitstring_generator(n):
        # do one "round" of updates on the vertex status
        for vertex in word:
            start_status[vertex] = 0
            start_status[vertex] = int(not np.any(np.logical_and(M[vertex], start_status)))
        # now check that the status vector is not updated on the second pass; if it is, then reject
        for vertex in np.arange(n, dtype=np.uint8):
            # only issues which can arise are when the vertex and its neighbors are all zero
            # - needn't set vertex to zero before eval
            if (start_status[vertex] == 0) and (not np.any(np.logical_and(M[vertex], start_status))):
                return False
    return True


@njit
def permis_count(M: np.ndarray):
    n = len(M)
    count = 0
    for word in ehrlich_permutation_generator(n):
        if is_permis(M, word, n):
            count += 1
    return count


@njit
def find_permis(M: np.ndarray):
    """
    :param M: adjacency matrix of graph G
    :return: np.array a permis for graph G if one exists, or np.zeros(n) otherwise.
    """
    n = M.shape[0]
    for word in ehrlich_permutation_generator(n):
        if is_permis(M, word):
            return word.copy()
    return np.zeros(n, dtype=np.uint8)


@njit
def find_permis_whp(M: np.ndarray, tries=1000):
    """
    Attempts to find a permis by trying ones chosen uniformly at random.
    :param tries: how many attempts should be made - typically 10<tries<5000
    :param M: adjacency matrix of graph G
    :return: np.array a permis for graph G if one exists, or np.zeros(n) otherwise.
    """
    n = M.shape[0]
    word = np.arange(n, dtype=np.uint8)
    if is_permis(M, word):
        return word
    for i in range(tries):
        word = np.random.permutation(word)
        if is_permis(M, word):
            return word.copy()
    return np.zeros(n, dtype=np.uint8)


@njit
def has_permis(M: np.ndarray):
    """
    :param M: adjacency matrix of graph G
    :return: 1 if G admits a permis, 0 otherwise
    """
    n = M.shape[0]
    for word in ehrlich_permutation_generator(n):
        if is_permis(M, word):
            return 1
    return 0


@njit
def has_permis_whp(M: np.ndarray, tries=500):
    """
    Attempts to find a permis by trying ones chosen uniformly at random.
    :param tries: how many attempts should be made - typically 10<tries<5000
    :param M: adjacency matrix of graph G
    :return: 1 if G admits a permis, 0 otherwise
    """
    n = M.shape[0]
    word = np.arange(n, dtype=np.uint8)
    if is_permis(M, word):
        return 1
    for i in range(tries):
        word = np.random.permutation(word)
        if is_permis(M, word):
            return 1
    return 0


@njit  # todo: implement in such a way that parallelism isn't squandered - atm everyone waits for the permisless graph
def has_permis_hybrid_BAD(M):
    """
    First implementation of hybrid run, but squanders parallelism - only a few threads end up running has_permis(M),
    and those dominate runtime. DO NOT USE EXCEPT FOR COMPARISON.
    :param M: adjacency matrix
    :return: 1 if G admits a permis, 0 otherwise
    """
    lucky_hit = has_permis_whp(M)
    if lucky_hit:
        return lucky_hit
    return has_permis(M)
