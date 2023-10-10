import time
from g6_reader import graph6_to_numpy_stack
import networkx as nx
import numpy as np


def oeis_A001349(n):
    """
    Reference: https://oeis.org/A001349
    :param n: number of vertices
    :returns: the number of connected graphs on n vertices
    """
    arr = [1, 1, 1, 2, 6, 21, 112, 853, 11117, 261080, 11716571,
           1006700565, 164059830476, 50335907869219,
           29003487462848061, 31397381142761241960,
           63969560113225176176277,
           245871831682084026519528568,
           1787331725248899088890200576580,
           24636021429399867655322650759681644]
    return arr[n]


def is_cycle(M):
    # if some vertex has degree != 2 then M cannot be a cycle
    if np.any(M.sum(axis=0) != 2):
        return False
    # check cycle size by doing a walk from vertex 0 along component; if 0 is reached in <n steps then G is disconnected
    n = M.size[0]
    prev = None
    current = 0
    for _ in range(n - 1):
        neigh1, neigh2 = np.where(M[current])[0]  # point to the two neighbors of vertex 0
        if neigh1 == prev:
            next_v = neigh2
        else:
            next_v = neigh1
        prev = current
        current = next_v
        if current == 0:  # vertex 0 was reached in <= n-1 steps
            return False
    return True


def nx_loader(n):
    return nx.read_graph6(f'geng_outputs/graph{n}c.g6')


def optimized_loader(n):
    return graph6_to_numpy_stack(f'geng_outputs/graph{n}c.g6')


def matrices_for(Gs):
    Ms = [nx.to_numpy_array(G) for G in Gs]
    return Ms


def np_mat_for(Gs):
    n = len(Gs[0].nodes)
    num_graphs = len(Gs)
    Ms = np.empty((num_graphs, n, n))
    for i in range(num_graphs):
        Ms[i] = nx.to_numpy_array(Gs[i])
    return Ms


def test_mat_conversion(n, reps=10):
    Gs = nx_loader(n)
    for mat_for in [np_mat_for, matrices_for]:
        print(f"Evaluating {mat_for.__name__}")
        times = []
        for it in range(reps):
            start = time.perf_counter()
            Ms = mat_for(Gs)
            stop = time.perf_counter()
            print(f"Finished batch {n} in {stop - start:3f} seconds")
            times.append(stop - start)
            print(len(Ms), np.sum(Ms))
    print(f"Run takes in {np.mean(times):3f} ± {np.std(times):3f} seconds")


def test_g6_load(n, reps=10):
    print(f"Evaluating g6 load time")
    times = []
    # Get numba jitted

    print("Warmed up:", Gs)
    for loader in [nx_loader, optimized_loader]:
        print(f"---Evaluating loader {loader.__name__}---")
        for it in range(reps):
            start = time.perf_counter()
            Gs = loader(n)
            stop = time.perf_counter()
            print(f"Finished batch {n} in {stop - start:3f} seconds")
            times.append(stop - start)
            print(len(Gs))
        print(f"Run takes {np.mean(times):3f} ± {np.std(times):3f} seconds")
        print()


if __name__ == '__main__':
    test_g6_load(10, reps=3)
