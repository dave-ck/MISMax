# Original author: D. Eppstein, UC Irvine, August 12, 2003.
# The original code at http://www.ics.uci.edu/~eppstein/PADS/ is public domain.
# Modified (badly and also a lot) by D. Kutner, Durham University, July 27, 2023
"""Functions for reading and writing graphs in the *graph6* format.

The *graph6* file format is suitable for small graphs or large dense
graphs. For large sparse graphs, use the *sparse6* format.

For more information, see the `graph6`_ homepage.

.. _graph6: http://users.cecs.anu.edu.au/~bdm/data/formats.html

DKutner: changed to specifically read nauty/geng outputs into a large
 numpy 3D array very efficiently; stripped out a lot of error
  checking, used numba's njit, and replaced lists with np.arrays
   where easily feasible.
"""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numba import njit


@njit
def numpy_adjmatrix_from_g6_bytes(bytes_in):
    data = np.array([c - 63 for c in bytes_in])
    n, data = data_to_n(data)
    M = np.zeros((n, n), dtype=np.uint8)
    for (i, j), b in zip([(i, j) for j in range(1, n) for i in range(j)], bits(data)):
        M[i, j] = b
        M[j, i] = b
    return M


@njit
def bits(data):
    for d in data:
        for i in np.arange(5, -1, -1):
            yield (d >> i) & 1


def graph6_to_numpy_stack(path):
    """
    Graph6 specification <http://users.cecs.anu.edu.au/~bdm/data/formats.html>
    """
    with open(path, "rb") as file:
        # first, get a count of how many graphs there are
        num_graphs = 0
        for line in file:
            num_graphs += 1
        # then recover n from the last line
        n = numpy_adjmatrix_from_g6_bytes(line).shape[0]
    with open(path, "rb") as file:
        Ms = np.zeros((num_graphs, n, n), dtype=np.uint8)
        i = 0
        for line in file:
            line = line.strip()
            if not len(line):
                continue
            Ms[i] = numpy_adjmatrix_from_g6_bytes(line)
            i += 1
        return Ms


def graph_generator(path):
    """
    Graph6 specification <http://users.cecs.anu.edu.au/~bdm/data/formats.html>
    """
    with open(path, "rb") as file:
        i = 0
        for line in file:
            line = line.strip()
            if not len(line):
                continue
            yield numpy_adjmatrix_from_g6_bytes(line)
            i += 1


@njit
def data_to_n(data):
    """Read initial one-, four- or eight-unit value from graph6
    integer sequence.

    Return (value, rest of seq.)"""
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    return (
        (data[2] << 30)
        + (data[3] << 24)
        + (data[4] << 18)
        + (data[5] << 12)
        + (data[6] << 6)
        + data[7],
        data[8:],
    )


if __name__ == '__main__':
    print("hi")
    Ms = graph6_to_numpy_stack("geng_outputs/graph4c.g6")
    for M in Ms:
        G = nx.from_numpy_array(M)
        plt.title(f"G: {G}")
        print(G)
        nx.draw(G)
        plt.show()
