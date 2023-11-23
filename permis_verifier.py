import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from permis import is_permis, find_permis_whp
from g6_reader import graph6_to_numpy_stack
from numba import njit
from graph_utils import has_induced_c_k, is_cycle


def verify_permises_for(n):
    permises = np.load(f"permis_tables/permises_for_g{n}c.npy")
    adj_matrices = graph6_to_numpy_stack(f"./geng_outputs/graph{n}c.g6")
    return verify_permises(adj_matrices, permises, n)


@njit
def verify_permises(adj_matrices, permises, n):
    count = 0
    for adj_matrix, permis in zip(adj_matrices, permises):
        if not is_permis(adj_matrix, permis):
            print("Rejected permis number", count, ":", permis)
            impossible_permis = find_permis_whp(adj_matrix, tries=500)
            if np.any(impossible_permis):
                print("Lies????")
                print(impossible_permis)
                print(adj_matrix)
            count += 1
    print("Verified all", n, "vertex graphs")


def verify_induced_odd_holes_and_antiholes(n):
    permises = np.load(f"permis_tables/permises_for_g{n}c.npy")
    adj_matrices = graph6_to_numpy_stack(f"./geng_outputs/graph{n}c.g6")
    count = 0
    id = 0
    for adj_matrix, permis in zip(adj_matrices, permises):
        if not np.any(permis):
            complement_adj_matrix = np.ones_like(adj_matrix) - adj_matrix - np.eye(n, dtype=np.uint8)
            perfect = True
            for k in [5, 7, 9]:
                if has_induced_c_k(complement_adj_matrix, k):
                    print(f"Graph {count} (id {id}) with no permis has an induced anti-C_{k}")
                    perfect = False
                if has_induced_c_k(adj_matrix, k):
                    print(f"Graph {count} (id {id}) with no permis has an induced C_{k}")
                    perfect = False

            if perfect:
                print(f"Graph {count} (id {id}) with no permis is perfect!! ------------------------------")
                G = nx.from_numpy_array(adj_matrix)
                nx.draw(G)
                plt.savefig(f"./py_outputs/permisless_perfect_graphs/{n}_vertex_permisless_perfect_graph_{id}.png")
                plt.show()
            print()
            count += 1
        id += 1
    print(f"Total {count} permisless {n} vertex graphs.")


if __name__ == '__main__':
    # for n in range(3, 10):
    #     verify_permises_for(n)
    verify_induced_odd_holes_and_antiholes(9)
