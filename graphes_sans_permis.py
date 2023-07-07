import itertools
from time import time

import networkx as nx
import numpy as np
from numba import jit

# outputs were generated with geng from the nauty package: https://pallini.di.uniroma1.it/
from matplotlib import pyplot as plt


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


# exhaustively searches all graphs in the graph atlas - only C_7 does not admit a permis
def exhaust_nx_smallgraphs():
    for G in nx.graph_atlas_g():
        G_good, witness = is_good_adj_matrix(nx.to_numpy_array(G))
        if G_good:
            print("Graph", G, "has permis", witness)
        else:
            print("Graph", G, "has no permis")  # occurs only for C_7, G353
            nx.draw(G)
            plt.show()


def exhaust_n_vertex_connected_graphs(n):
    i = 0
    Gs = connected_graphs_on(n)
    print("Graphs on %d vertices loaded from .graph6 file. Computing..." %n )
    for G in Gs:
        G_good, witness = is_good_adj_matrix(nx.to_numpy_array(G))
        if G_good:
            print("G %d was good with witness %s" % (i, witness))
            pass
        else:
            plt.title("G was bad")
            print("G was bad")
            folder = "C:/Users/Administrator/OneDrive - Durham University/MISMax/bad_%d/" % n
            nx.draw(G, with_labels=True)
            plt.savefig(folder + "%d_vertex_graph_%d.png" % (n, i))
            plt.show()
        i += 1


start = time()
exhaust_n_vertex_connected_graphs(7)
print("Finished batch 7 in", time() - start, "seconds")
exhaust_n_vertex_connected_graphs(8)
print("Finished batch 8 in", time() - start, "seconds")
exhaust_n_vertex_connected_graphs(9)
print("Finished batch 9 in", time() - start, "seconds")