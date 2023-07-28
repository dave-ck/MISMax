from permis import has_permis, find_permis, find_permis_whp, has_permis_hybrid_BAD
from g6_reader import adj_matrix_generator
from generators import filter_generator
from graph_utils import oeis_A001349
# from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import numpy as np
from time import perf_counter


def classify_graphs_on(n, processes=16, chunksize=128):
    graph_iterable = adj_matrix_generator(f"./geng_outputs/graph{n}c.g6")
    with Pool(processes=processes) as pool:
        results = pool.imap(has_permis, graph_iterable, chunksize=chunksize)
        num_graphs = oeis_A001349(n)
        out_arr = np.empty(num_graphs, dtype=np.uint8)
        for res, i in zip(results, range(num_graphs)):
            out_arr[i] = res
    return out_arr


def get_permises(n, processes=16, chunksize=128):
    graph_iterable = adj_matrix_generator(f"./geng_outputs/graph{n}c.g6")
    with Pool(processes=processes) as pool:
        results = pool.imap(find_permis, graph_iterable, chunksize=chunksize)
        num_graphs = oeis_A001349(n)
        out_arr = np.empty((num_graphs, n), dtype=np.uint8)
        for res, i in zip(results, range(num_graphs)):
            out_arr[i] = res
    return out_arr


def test_time(func, n=7, processes=32, chunksize=256, reps=10):
    times = []
    counts = []
    print(f"\n--- Running function {func.__name__} with n={n}, {processes} processes, {chunksize} chunksize,"
          f" including generator initialization ---")
    with Pool(processes=processes) as pool:
        for it in range(reps + 1):
            start = perf_counter()
            graph_iterable = adj_matrix_generator(f"./geng_outputs/graph{n}c.g6")
            result = pool.imap(func, graph_iterable, chunksize=chunksize)
            # must do something with results, otherwise they won't be evaluated because imap is lazy
            # here, just sum() because we are interested in comparing the time to compute the results
            count = sum(result)
            stop = perf_counter()
            print(f"Finished run {it:2} with {n} vertex graphs in {stop - start:.3f} seconds, found {count} permises")
            if it > 0:
                # exclude first run due to compilation overhead etc.
                times.append(stop - start)
                counts.append(count)

    print(f"Run takes {np.mean(times):.3f} ± {np.std(times):.3f} seconds,"
          f" finds {np.mean(counts):.0f} ± {np.std(counts):.0f} permises")


def hybrid_permis_finder(n, processes=32, chunksize=256):
    print(f"\n--- Running hybrid permis finder with n={n}, {processes} processes, {chunksize} chunksize ---")
    with Pool(processes=processes) as pool:
        num_graphs = oeis_A001349(n)
        permis_table = np.zeros((num_graphs, n), dtype=np.uint8)
        start = perf_counter()
        graph_iterable = adj_matrix_generator(f"./geng_outputs/graph{n}c.g6")
        result = pool.imap(find_permis_whp, graph_iterable, chunksize=chunksize)
        count = 0
        for permis, i in zip(result, range(num_graphs)):
            permis_table[i] = permis
            count += np.any(permis)
        stop = perf_counter()
        print(f"Random run finished in {stop - start:.3f} seconds, {num_graphs - count} "
              f"graphs are permisless whp; {count} permises found")
        # fresh graph iterable; each one runs through file once, need to reset to file start
        graph_iterable = adj_matrix_generator(f"./geng_outputs/graph{n}c.g6")
        # find which graphs have no permis with high probability (because we haven't found one yet)
        whp_permisless = (np.sum(permis_table, axis=1) == 0)
        whp_permisless_graphs = filter_generator(graph_iterable, whp_permisless)
        result = pool.imap(find_permis, whp_permisless_graphs, chunksize=1)  # use chunksize 1 - each task expensive
        whp_permisless_indices = np.where(whp_permisless)[0]
        for (permis, index) in zip(result, whp_permisless_indices):
            permis_table[index] = permis
            count += np.any(permis)
        stop = perf_counter()
        np.save(f"permis_tables/permises_for_g{n}c.npy", permis_table)
        print(f"Finished update of permis_table for {n} vertex graphs in {stop - start:.3f} seconds,"
              f" {num_graphs - count} graphs are permisless; {count} permises found")

if __name__ == "__main__":
    # test_time(has_permis_hybrid_BAD, n=8)
    for n in range(3, 10):
        hybrid_permis_finder(n)
    print("Done.")
