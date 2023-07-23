import itertools
import math
from time import time

import networkx as nx
import numpy as np
from numba import jit, njit, prange
from multiprocessing.pool import ThreadPool

# outputs were generated with geng from the nauty package: https://pallini.di.uniroma1.it/
from matplotlib import pyplot as plt

pool = ThreadPool()
THREAD_COUNT = 32
bitstrings_cache = {}


def connected_graphs_on(n):
    return nx.read_graph6('geng_outputs/connected_%d_vertices.graph6' % n)


def numpy_bitstrings(n):
    out_arr = np.zeros((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        bitstring = np.binary_repr(i, width=n)
        out_arr[i] = np.array(list(bitstring), dtype=np.uint8)
    return out_arr


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


@jit(nopython=True)
def has_permis(M, start_status_array, perms_n):
    n = len(M)
    # here sequential processing is better - if lucky we get a boost because something
    # is accepted with an "early" word e.g. (0, 1, 3, 4, 5, 6, 2)
    for word in perms_n:
        if is_permis(M, word, start_status_array.copy(), n):
            return True
    return False


def adj_matrix_permis_count(M, start_status_array):
    n = len(M)
    count = 0
    for word in permutations(n):
        if good_under_perm_(M, word, start_status_array.copy(), n):
            count += 1
    return count


@jit(nopython=True)
def is_permis(M, word, start_status_array, n):
    for i in range(0, 2 ** n):
        start_status = start_status_array[i]
        # do one "round" of updates on the vertex status
        for vertex in word:
            start_status[vertex] = np.logical_not(np.any(np.logical_and(M[vertex], start_status)))
        # now check that the status vector is not updated on the second pass; if it is, then reject
        for vertex in word:
            # only issues which can arise are when the vertex and its neighbors are all zero
            # - needn't set vertex to zero before eval
            if (start_status[vertex] == 0) and (np.logical_not(np.any(np.logical_and(M[vertex], start_status)))):
                return False
    return True


# exhaustively searches all graphs in the graph atlas - only C_7 does not admit a permis
def exhaust_nx_smallgraphs():
    for G in nx.graph_atlas_g():
        n = G.number_of_nodes()
        G_good, witness = has_permis(nx.to_numpy_array(G),
                                     numpy_bitstrings(G.number_of_nodes()),
                                     permutations(n))
        if G_good:
            print("Graph", G, "has permis", witness)
        else:
            print("Graph", G, "has no permis")  # occurs only for C_7, G353
            nx.draw(G)
            plt.show()


# writing good graphs is very expensive; plotting graphs is resource-intensive, but also useful to infer information
def exhaust_n_vertex_connected_graphs(n, plot=True, write_g6=False):
    i = 0
    Gs = connected_graphs_on(n)[6200:6300]  # [34544:34544 + 32]  # todo: remove me
    print("Graphs on %d vertices loaded from .graph6 file. Computing adj matrix stack..." % n)
    num_graphs = len(Gs)
    Ms = np.zeros([num_graphs, n, n], dtype=np.uint8)
    for i in range(num_graphs):
        Ms[i] = nx.to_numpy_array(Gs[i])
    print("Calling Ms_good on stack.")
    is_good = Ms_good(Ms, n, num_graphs)
    for i in range(num_graphs):
        G_good, witness = is_good[i], None  # is_good_adj_matrix(Ms[i], start_status_array)
        if not G_good:
            # folder = "C:/Users/Administrator/OneDrive - Durham University/MISMax/bad_%d/" % n
            folder = "py_outputs/bad_%d/" % n
            if plot:
                plt.title("G was bad")
                print("G was bad")
                nx.draw(Gs[i], with_labels=True)
                plt.savefig(folder + "%d_vertex_graph_%d.png" % (n, i))
                plt.show()
            if write_g6:
                nx.write_graph6(Gs[i], folder + "%d_%d.g6" % (n, i))
        i += 1


def Ms_good(Ms, n, num_graphs):
    perms_n = permutations(n)
    start_status_array = numpy_bitstrings(n)
    return Ms_good_starmap(Ms, num_graphs, start_status_array, perms_n)


@njit
def Ms_good_sequential(Ms, num_graphs, start_status_array, perms_n):
    is_good = np.zeros(num_graphs, dtype=np.uint8)
    for i in range(num_graphs):
        is_good[i] = has_permis(Ms[i], start_status_array.copy(), perms_n.copy())
    return is_good


@njit(parallel=True)
def Ms_good_prange(Ms, num_graphs, start_status_array, perms_n):
    is_good = np.zeros(num_graphs, dtype=np.uint8)
    BATCH_SIZE = num_graphs / THREAD_COUNT
    for j in prange(THREAD_COUNT):
        for i in range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE):
            is_good[i] = has_permis(Ms[i], start_status_array.copy(), perms_n.copy())
    return is_good


@njit
def make_starmap_params(Ms, num_graphs, start_status_array, perms_n):
    params = [(Ms[i], start_status_array.copy(), perms_n.copy()) for i in np.arange(num_graphs)]
    return params


def Ms_good_starmap(Ms, num_graphs, start_status_array, perms_n):
    params = make_starmap_params(Ms, num_graphs, start_status_array,
                                 perms_n)  # [(Ms[i], start_status_array.copy(), perms_n.copy()) for i in np.arange(num_graphs)]
    return pool.starmap(has_permis, params, chunksize=1000)


def testing():
    n = 7
    Gs = connected_graphs_on(n)
    num_graphs = len(Gs)
    Ms = np.zeros([num_graphs, n, n], dtype=np.uint8)
    for i in range(num_graphs):
        Ms[i] = nx.to_numpy_array(Gs[i])
    perms_n = permutations(n)
    start_status_array = numpy_bitstrings(n)
    # Run test 5 times to see if it was a fluke

    print("Warmup and prereqs complete.")
    for Ms_good_ in [Ms_good_starmap,
                     Ms_good_sequential,
                     Ms_good_prange]:
        print("-------", Ms_good_.__name__, "-------")

        start = time()
        is_good = Ms_good_(Ms, num_graphs, start_status_array, perms_n)
        for i in range(num_graphs):
            G_good = is_good[i]
            if not G_good:
                plt.title("G%d was bad" % i)
                print("G%d was bad" % i)
                nx.draw(Gs[i], with_labels=True)
                plt.show()
            i += 1

        stop = time()
        print("Finished warmup in", stop - start, "seconds")

        times = []

        # Run test 5 times to see if it was a fluke
        for it in range(5):
            start = time()

            is_good = Ms_good_(Ms, num_graphs, start_status_array, perms_n)
            for i in range(num_graphs):  # here slice chosen for n=8, to hit bad cluster
                G_good = is_good[i]
                if not G_good:
                    plt.title("G%d was bad" % i)
                    print("G%d was bad" % i)
                    nx.draw(Gs[i], with_labels=True)
                    plt.show()
                i += 1

            stop = time()
            print("Finished batch", it, "in", stop - start, "seconds")
            times.append(stop - start)
        print("Computed in %f ± %f seconds" % (np.mean(times), np.std(times)))


testing()
