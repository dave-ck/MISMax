import numpy as np
from numba import njit
from time import time, perf_counter


@njit
def shifty_bitstring_generator(n):
    A = np.empty(n, dtype=np.uint8)
    for i in range(2 ** n):
        for j in range(n - 1, -1, -1):
            A[j] = i & 1
            i >>= 1
        # A contains the binary representation of i
        yield A
    return


def bitwise_iter(n, _):
    A = np.empty(n, dtype=np.uint8)
    a = 0
    for i in range(2 ** n):
        temp = i
        for j in range(n - 1, -1, -1):
            A[j] = i & 1
            i >>= 1
        # A contains the binary representation of i
        a += do_stuff(A)
    return a


@njit
def bitwise_iter_numba(n, _):
    A = np.empty(n, dtype=np.uint8)
    a = 0
    for i in range(2 ** n):
        for j in range(n - 1, -1, -1):
            A[j] = i & 1
            i >>= 1
        # A contains the binary representation of i
        a += do_stuff(A)
    return a


def memory_iter(n, precomputed_array):
    a = 0
    for i in range(2 ** n):
        a += do_stuff(precomputed_array[i])
    return a


def generate_with_binary_repr(n):
    out_arr = np.empty((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        bitstring = np.binary_repr(i, width=n)
        out_arr[i] = np.array(list(bitstring), dtype=np.uint8)
    return out_arr


def generate_with_bitwise(n):
    out_arr = np.empty((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        index_i = i
        for j in range(n - 1, -1, -1):
            out_arr[index_i, j] = i & 1
            i >>= 1
    return out_arr


def generate_with_bitwise_toms_way(n):
    out_arr = np.empty((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        index_i = i
        for j in range(n - 1, -1, -1):
            out_arr[index_i, j] = i & 1
            i >>= 1
    return out_arr


@njit
def generate_with_bitwise_toms_way_numba(n):
    out_arr = np.empty((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        index_i = i
        for j in range(n - 1, -1, -1):
            out_arr[index_i, j] = i & 1
            i >>= 1
    return out_arr


@njit
def generate_with_bitwise_numba(n):
    out_arr = np.empty((2 ** n, n), dtype=np.uint8)
    for i in range(2 ** n):
        index_i = i
        for j in range(n - 1, -1, -1):
            out_arr[index_i, j] = i & 1
            i >>= 1
    return out_arr


@njit
def do_stuff(A):
    return np.sum(A)


def test_generators(n):  # times for n=20 on Ryzen 7 5800H in comment; for smaller n numba appears as zero
    for generate in [generate_with_bitwise_numba,  # 0.015 ± 0.000 seconds
                     generate_with_bitwise,  # 2.788 ± 0.080 seconds
                     generate_with_binary_repr,  # 3.127 ± 0.107 seconds
                     ]:
        print(generate.__name__, "with n=", n)
        # Compile and warmup JIT
        for _ in range(5):
            generate(3)
        times = []
        # Run test 10 times
        for _ in range(10):
            # Run function and measure time
            start_time = perf_counter()
            result = generate(n)
            elapsed_time = perf_counter() - start_time

            print(f"{elapsed_time:.3f} seconds - result sum: {np.sum(result)}")

            if n <= 4:
                print(result)  # for small n, can print result to verify correctness
            times.append(elapsed_time)
        print(f"Computed in {np.mean(times):.3f} ± {np.std(times):.3f} seconds")
        print()


def test_iterators(n, reps):  # times for n=10, reps=1000 on Ryzen 7 5800H in comment
    # precompute bitstrings
    precomputed_array = generate_with_binary_repr(n)
    for iterate in [bitwise_iter,  # 1.454 ± 0.016 seconds
                    bitwise_iter_numba,  # 0.015 ± 0.000 seconds
                    memory_iter,  # 0.294 ± 0.003 seconds
                    ]:
        print(iterate.__name__, "with n=", n)
        # Compile and warmup JIT
        iterate(n, precomputed_array)

        times = []
        # Run test 10 times
        for _ in range(10):
            # Run function and measure time
            start_time = perf_counter()
            for rep in range(reps):
                result = iterate(n, precomputed_array)
            elapsed_time = perf_counter() - start_time

            print(f"{elapsed_time:.3f} seconds - result sum: {np.sum(result)}")

            if n <= 4:
                print(result)  # for small n, can print result to verify correctness
            times.append(elapsed_time)
        print(f"Computed in {np.mean(times):.3f} ± {np.std(times):.3f} seconds")
        print()


# test_generators(20)
test_iterators(20, 1)
