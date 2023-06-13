from MISMax import numpy_bitstrings, is_min_under_cyclic_shifts

def cyclic_testing():
    for bitstring in numpy_bitstrings(13):
        if is_min_under_cyclic_shifts(bitstring):
            print(bitstring, "is min")
        else:
            print(bitstring, "is not min")

cyclic_testing()