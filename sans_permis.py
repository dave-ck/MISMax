import itertools
import math

import networkx as nx
import numpy as np


def numpy_bitstrings(n):
    out_arr = np.zeros((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        bitstring = np.binary_repr(i, width=n)
        out_arr[i] = np.array(list(bitstring), dtype=np.uint8)
    return out_arr


def is_good(G):
    n = len(G.nodes)
    for perm in itertools.permutations([i for i in range(n)]):
        if good_under_perm(G, perm):
            return True, perm
    return False, None


def good_under_perm(G, perm):
    n = len(G.nodes)
    M = nx.to_numpy_array(G, dtype=np.uint8)
    for starting_status in numpy_bitstrings(n):
        status = starting_status
        # do one "round" of updates on the vertex status
        for vertex in perm:
            status[vertex] = 0
            status[vertex] = int(not np.any(np.logical_and(M[vertex], status)))
        # now check that the status vector is not updated on the second pass; if it is, then reject
        for vertex in perm:
            # only issues which can arise are when the vertex and its neighbors are all zero - needn't set vertex to zero before eval
            if (status[vertex] == 0) and (not np.any(np.logical_and(M[vertex], status))):
                return False
        # if reached: ok for this starting status, go to the next
    return True


def classify_atlas():
    for G in nx.graph_atlas_g():
        G_good, witness = is_good(G)
        if G_good:
            print("Graph", G, "has permis", witness)
        else:
            print("Graph", G, "has no permis")  # occurs only for C_7, G353


classify_atlas()
