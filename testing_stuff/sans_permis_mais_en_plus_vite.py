import itertools
import math
import time
from os import getpid
import networkx as nx
import numpy as np
import pandas as pd
from numba import jit, njit, prange
from matplotlib import pyplot as plt
from multiprocessing import Pool


# geng_outputs were generated with geng from the nauty package: https://pallini.di.uniroma1.it/

def connected_graphs_on(n):
    return nx.read_graph6(f'geng_outputs/graph{n}c.g6')


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


# from pseudocode in Knuth TAOCP vol 4 Algorithm E
@njit
def ehrlich_permutation_generator(n):
    a = np.arange(n)
    b = np.arange(n)
    c = np.zeros(n + 1)  # c[0] should never be accessed; using arr of length n+1 to index starting at 1
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


@njit
def is_good_adj_matrix(M):
    n = len(M)
    # here sequential processing is better - if lucky we get a boost because something
    # is accepted with an "early" word e.g. (0, 1, 3, 4, 5, 6, 2)
    for word in ehrlich_permutation_generator(n):
        if good_under_perm_(M, word, n):
            return True  # word
    return False  # , None


@njit
def adj_matrix_permis_count(M):
    n = len(M)
    count = 0
    for word in ehrlich_permutation_generator(n):
        if good_under_perm_(M, word, n):
            count += 1
    return count


@njit
def good_under_perm_(M, word, n):
    for start_status in shifty_bitstring_generator(n):
        # start_status = start_status_array[i]
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


def Ms_good_batch(Ms, pool, chunksize=32):
    num_graphs, n, _ = Ms.shape
    if not pool:
        with Pool(processes=processes) as pool:
            res = pool.map(is_good_adj_matrix, Ms, chunksize=chunksize)
    else:
        res = pool.map(is_good_adj_matrix, Ms, chunksize=chunksize)
    return res


# writing good graphs is very expensive; plotting graphs is resource-intensive, but also useful to infer information
def exhaust_n_vertex_connected_graphs(n, pool, chunksize=None, plot=True, write_g6=False):
    Gs = connected_graphs_on(n)
    num_graphs = len(Gs)
    Ms = np.zeros([num_graphs, n, n], dtype=np.uint8)
    for i in range(num_graphs):
        Ms[i] = nx.to_numpy_array(Gs[i])
    isgood_vector = Ms_good_batch(Ms, pool, chunksize)
    for i in range(num_graphs):
        G_good = isgood_vector[i]
        if not G_good:
            folder = "./junk_outputs/"
            if plot:
                plt.title("G was bad")
                nx.draw(Gs[i], with_labels=True)
                plt.savefig(folder + "%d_vertex_graph_%d.png" % (n, i))
                plt.show()
            if write_g6:
                nx.write_graph6(Gs[i], folder + "%d_%d.g6" % (n, i))
        i += 1


def do_some_testing_I_guess(n, reps=10, chunksize=32, processes=32):
    times = []
    with Pool(processes=processes) as pool:
        # get numba to compile stuff
        exhaust_n_vertex_connected_graphs(n, chunksize=chunksize, pool=pool, plot=True,
                                          write_g6=True)
        for it in range(reps):
            start = time.perf_counter()
            exhaust_n_vertex_connected_graphs(n, chunksize=chunksize, pool=pool, plot=True,
                                              write_g6=True)
            stop = time.perf_counter()
            print(f"Finished batch {n} in {stop - start:3f} seconds")
            times.append(stop - start)
    print(f"Run takes in {np.mean(times):3f} ± {np.std(times):3f} seconds")


def counts_for_atlas(n):
    graph_i = 0
    nfac = math.factorial(n)
    counts = []
    rates = []
    for G in connected_graphs_on(n):
        count = adj_matrix_permis_count(nx.to_numpy_array(G))
        rate = count / nfac
        print(f"The permis count for graph graph{n}c[{graph_i}] is: {count} out of {nfac} ({100 * count / nfac:3f}%)")
        rates.append(rate)
        counts.append(count)
        graph_i += 1
    df_out = pd.DataFrame(
        {
            "Graph_i": np.arange(graph_i, dtype=np.uint8),
            "Count": counts,
            "Rate": rates
        })
    df_out.to_csv(f'permis_counts/permis_counts_graph{n}c.csv')


if __name__ == "__main__":
    processes = 8
    chunksize = 16
    print(f"---Running with {processes} processes and chunksize {chunksize}: ---")
    do_some_testing_I_guess(7, chunksize=chunksize, processes=processes)
    print()
