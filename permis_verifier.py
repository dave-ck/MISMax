import numpy as np
from permis import is_permis, find_permis_whp
from g6_reader import graph6_to_numpy_stack
from numba import njit


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


if __name__ == '__main__':
    for n in range(3, 10):
        verify_permises_for(n)
