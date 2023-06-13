import itertools

from time import time
import networkx as nx
import numpy as np
import numba
from matplotlib import pyplot as plt
from numba import jit, njit, prange

BATCH_SIZE = 32  # must be a power of 2


def my_graph():
    G = nx.cycle_graph(5)
    # could pass any networkx graph; here C5 for testing purposes
    return G


def numpy_bitstrings(n):
    out_arr = np.zeros((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        bitstring = np.binary_repr(i, width=n)
        out_arr[i] = np.array(list(bitstring), dtype=np.uint8)
    return out_arr


# memoizing the below gives only a negligible performance boost (for small n - as in our case)
# from https://newbedev.com/generating-all-permutations-efficiently
def permutations(n):
    a = np.zeros((np.math.factorial(n), n), np.uint8)
    f = 1
    for m in range(2, n + 1):
        b = a[:f, n - m + 1:]  # the block of permutations of range(m-1)
        for i in range(1, m):
            a[i * f:(i + 1) * f, n - m] = i
            a[i * f:(i + 1) * f, n - m + 1:] = b + (b >= i)
        b += 1
        f *= m
    return a


def is_good_adj_matrix(M):
    n = len(M)
    # here sequential processing is better - if lucky we get a boost because something
    # is accepted with an "early" word e.g. (0, 1, 3, 4, 5, 6, 2)
    start_status_array = numpy_bitstrings(n)
    for word in permutations(n):
        if good_under_perm_(M, word, start_status_array.copy(), n):
            return True, word
    return False, None


@jit(nopython=True)
def good_under_perm_(M, word, start_status_array, n):
    for i in range(0, 2 ** n):
        start_status = start_status_array[i]
        # do one "round" of updates on the vertex status
        for vertex in word:
            start_status[vertex] = 0
            start_status[vertex] = int(not np.any(np.logical_and(M[vertex], start_status)))
        # now check that the status vector is not updated on the second pass; if it is, then reject
        for vertex in word:
            # only issues which can arise are when the vertex and its neighbors are all zero
            # - needn't set vertex to zero before eval
            if (start_status[vertex] == 0) and (not np.any(np.logical_and(M[vertex], start_status))):
                return False
    return True


@jit(nopython=True)
def perm_fixes_from_start(M, word, start_status):
    # do one "round" of updates on the vertex status
    for vertex in word:
        start_status[vertex] = 0
        start_status[vertex] = int(not np.any(np.logical_and(M[vertex], start_status)))
    # now check that the status vector is not updated on the second pass; if it is, then reject
    for vertex in word:
        # only issues which can arise are when the vertex and its neighbors are all zero
        # - needn't set vertex to zero before eval
        if (start_status[vertex] == 0) and (not np.any(np.logical_and(M[vertex], start_status))):
            return 0
    return 1


@jit(nopython=True)
def is_min_under_cyclic_shifts(starting_bitstring):
    current = starting_bitstring
    while True:
        current = np.roll(current, shift=1)
        if np.all(current == starting_bitstring):
            return True
        if lex_less(current, starting_bitstring):
            return False


# return True if y<x lexicographically
@jit(nopython=True)
def lex_less(x, y):
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
        if y[i] > x[i]:
            return True


def main():
    start = time()
    #
    # n = 7
    # Cn = nx.cycle_graph(n)
    # pos = nx.circular_layout(Cn)
    # pos[n] = (0, 0)
    # for bitstring in numpy_bitstrings(n):
    #     if not is_min_under_cyclic_shifts(bitstring):
    #         print("Skipping graph-generating bitstring", bitstring, "; is_min_under_cyclic_shifts==False")
    #         continue
    #     G = Cn.copy()
    #     G.add_node(n)
    #     # decide depending on bitstring whether it is adjacent to each other node
    #     for i in range(n):
    #         if bitstring[i]:
    #             G.add_edge(i, n)
    #     G_good, witness = is_good(G)
    #
    #     if G_good:
    #         plt.title("With bitstring %s G was good with witness %s" % (bitstring, witness))
    #         folder = "C:/Users/Administrator/OneDrive - Durham University/MISMax/good/"
    #     else:
    #         plt.title("With bitstring %s G was bad" % bitstring)
    #         folder = "C:/Users/Administrator/OneDrive - Durham University/MISMax/bad/"
    #
    #     nx.draw(G, pos=pos, with_labels=True)
    #
    #     plt.savefig(folder + "Augmented_C%d_With_Bitstring_%s.png" % (n, bitstring))
    #     plt.show()

    for G in nx.graph_atlas_g()[350:360]:
        G_good, witness = is_good_adj_matrix(nx.to_numpy_array(G))
        if G_good:
            print("Good:", G, "with witness:", witness)
        else:
            print("Baad:", G, "with witness:", witness, "<-- should be None, for G353 *****")

    print("Run took: %5f seconds" % (time() - start))


main()
