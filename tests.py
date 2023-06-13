from MISMax import numpy_bitstrings, is_min_under_cyclic_shifts, is_good_adj_matrix
from time import time
import networkx as nx

def cyclic_testing():
    for bitstring in numpy_bitstrings(13):
        if is_min_under_cyclic_shifts(bitstring):
            print(bitstring, "is min")
        else:
            print(bitstring, "is not min")


def runtime_test():
    start = time()
    for G in nx.graph_atlas_g()[350:360]:
        G_good, witness = is_good_adj_matrix(nx.to_numpy_array(G))
        if G_good:
            print("Good:", G, "with witness:", witness)
        else:
            print("Baad:", G, "with witness:", witness, "<-- should be None, for G353 *****")

    print("Run took: %5f seconds" % (time() - start))


cyclic_testing()

