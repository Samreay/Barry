from mpi4py import MPI

import numpy as np


class Thingo:
    def __init__(self):
        pass

    def genpoint(self, x, y):
        return x * y

    def generate(self):
        global comm, size, rank
        xs = np.linspace(0, 10, 3)
        ys = np.linspace(100, 110, 3)

        all_indexes = [(i, j) for i in range(xs.size) for j in range(ys.size)]
        delegations = [all_indexes[i::size] for i in range(size)]

        run_indexes = comm.scatter(delegations, root=0)
        results = [self.genpoint(xs[i], ys[j]) for i, j in run_indexes]
        all_results = comm.gather(results, root=0)

        if rank == 0:
            data = np.empty((xs.size, ys.size))
            for d, r in zip(delegations, all_results):
                for (i, j), res in zip(d, r):
                    data[i, j] = res
            print(data)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    t = Thingo()
    t.generate()