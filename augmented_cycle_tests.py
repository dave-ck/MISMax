import itertools

import networkx as nx
import numpy as np
import numba
from numba import jit, njit, prange


def my_graph():
    G = nx.cycle_graph(7)
    nx.Graph.add_node()
    # could pass any networkx graph; here C5 for testing purposes
    return G


# using chatgpt
@jit(nopython=True)
def numpy_bitstrings(n):
    num_elements = 2 ** n
    array_shape = (num_elements, n)
    result_array = np.zeros(array_shape, dtype=np.uint8)

    for i in range(num_elements):
        bitstring = np.binary_repr(i, width=n)
        result_array[i] = np.array(list(bitstring), dtype=np.uint8)

    return result_array


def generate_bitstrings(n):
    if n <= 0:
        return [[]]
    result = []
    stack = [[1], [0]]
    while stack:
        current = stack.pop()
        if len(current) == n:
            result.append(current)
        else:
            stack.append(current + [1])  # Append 1 to the current bitstring
            stack.append(current + [0])  # Append 0 to the current bitstring
    return np.array(result, dtype=bool)


def is_good(G):
    n = len(G.nodes)
    possible_orderings = []
    for perm in itertools.permutations([i for i in range(n)]):
        if good_under_perm(G, perm):
            return True, perm
    return False, None


def good_under_perm(G, perm):
    M = nx.to_numpy_array(G, dtype=np.uint8)
    return good_under_perm_(M, perm)


@jit(nopython=True)
def good_under_perm_(M, perm):
    n = len(M)
    bitstrings = numpy_bitstrings(n)
    for i in range(2**n):
        starting_status = bitstrings[i]
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


G = nx.cycle_graph(7)
for G in nx.graph_atlas_g():
    G_good, witness = is_good(G)
    if G_good:
        print("Good:", G, "with witness:", witness)
    else:
        print("Baad:", G, "with witness:", witness, "<-- should be None *****")
