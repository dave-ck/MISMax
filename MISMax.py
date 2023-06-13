import itertools

import networkx as nx
import numpy as np


def my_graph():
    G = nx.cycle_graph(5)
    # could pass any networkx graph; here C5 for testing purposes
    return G


# chatgpt wrote this bit
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
    n = len(G.nodes)
    M = nx.to_numpy_array(G, dtype=np.uint8)
    for starting_status in generate_bitstrings(n):
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