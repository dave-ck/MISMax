import numpy as np
from numba import njit
from time import time, perf_counter


def ye_olde_matrix(n):
    a = np.zeros((np.math.factorial(n), n), np.uint8)
    f = 1
    for m in range(2, n + 1):
        b = a[:f, n - m + 1:]  # the block of permutations of range(m-1)
        for i in range(1, m):
            a[i * f:(i + 1) * f, n - m] = i
            a[i * f:(i + 1) * f, n - m + 1:] = b + (b >= i)
        b += 1
        f *= m
    return a


def ehrlich_gen(n):
    a = np.arange(n, dtype=np.uint8)
    b = np.arange(n, dtype=np.uint8)
    c = np.zeros(n + 1, dtype=np.uint8)  # c[0] should never be accessed; using arr of length n+1 to index starting at 1
    while True:
        yield a.copy()
        k = 1
        while c[k] == k:
            c[k] = 0
            k += 1
        if k == n:
            return  # is break better?
        c[k] += 1
        a[0], a[b[k]] = a[b[k]], a[0]
        # reversing sublist as in textbook - swap for numpy impl. later if compatible with njit
        j = 1
        k = k - 1
        while j < k:
            b[j], b[k] = b[k], b[j]
            j += 1
            k -= 1


# from pseudocode in Knuth TAOCP vol 4 Algorithm E
@njit
def ehrlich_gen_numba(n):
    a = np.arange(n)
    b = np.arange(n)
    c = np.zeros(n + 1)  # c[0] should never be accessed; using arr of length n+1 to index starting at 1
    while True:
        yield a.copy()
        k = 1
        while c[k] == k:
            c[k] = 0
            k += 1
        if k == n:
            return  # is break better?
        c[k] += 1
        a[0], a[b[k]] = a[b[k]], a[0]
        # reversing sublist as in textbook - swap for numpy impl. later if compatible with njit
        j = 1
        k = k - 1
        while j < k:
            b[j], b[k] = b[k], b[j]
            j += 1
            k -= 1


def ehrlich_matrix(n):
    perms = np.empty((np.math.factorial(n), n), dtype=np.uint8)
    i = 0
    for perm in ehrlich_gen(n):
        perms[i] = perm
        i += 1
    return perms


@njit
def ehrlich_matrix_numba(n):
    i = 0
    perms = np.zeros((factorial(n), n), dtype=np.uint8)
    for perm in ehrlich_gen_numba(n):
        # print(i, perm)
        perms[i] = perm
        i += 1
    return perms


@njit  # because for some reason math.factorial and np.math.factorial aren't supported by numba...
def factorial(n):
    out = 1
    for i in range(1, n + 1):
        out *= i
    return out


def test_generators(n):  # times for n=20 on Ryzen 7 5800H in comment; for smaller n numba appears as zero
    for generate in [ye_olde_matrix,  # 0.131 ± 0.004 seconds
                     ehrlich_matrix,  # 20.027 ± 0.648 seconds
                     ehrlich_matrix_numba,  # 1.017 ± 0.366 seconds
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


test_generators(10)
# print(ehrlich_matrix_numba(3))
