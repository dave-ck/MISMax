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


def verify_induced_odd_holes(n):
    permises = np.load(f"permis_tables/permises_for_g{n}c.npy")
    adj_matrices = graph6_to_numpy_stack(f"./geng_outputs/graph{n}c.g6")
    count = 0
    induced_c7_count = 0
    induced_c5_count = 0
    odd_hole_free_count = 0

    id = 0
    for adj_matrix, permis in zip(adj_matrices, permises):
        oddhole_found = False
        if not np.any(permis):
            if has_induced_c_k(adj_matrix, 5):
                print(f"Graph {count} (id {id}) with no permis has an induced C_5")
                oddhole_found = True
                induced_c5_count += 1

            if has_induced_c_k(adj_matrix, 7):
                print(f"Graph {count} (id {id}) with no permis has an induced C_7")
                oddhole_found = True
                induced_c7_count += 1

            if not oddhole_found and not has_induced_c_k(adj_matrix, 9):
                print(f"Graph {count} (id {id}) with no permis is induced odd hole free")
                odd_hole_free_count += 1

            count += 1
        id += 1
    print(f"Total {count} permisless {n} vertex graphs.")
    print(f"{induced_c5_count} graphs had an induced C_5, "
          f"{induced_c7_count} graphs had an induced C_7, and"
          f"{odd_hole_free_count} graphs had no induced odd hole.")


if __name__ == '__main__':
    # for n in range(3, 10):
    #     verify_permises_for(n)
    verify_induced_odd_holes(9)