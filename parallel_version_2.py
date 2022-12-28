from mpi4py import MPI
import numpy as np
import time
from threadpoolctl import threadpool_limits

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numproc = comm.Get_size()

num_of_iters = 100000

if rank == 0:

    with open("parameters.txt", 'r', encoding = 'utf-8') as f:
        dimension = np.array([np.int32(f.readline())])
        size = np.array([np.int32(f.readline())])
        hyperplane_parametrs = np.array([np.float64(i) for i in f.readline().split()])  

    ave, res = np.divmod(dimension, numproc)
    rcounts = np.empty(numproc, dtype=np.int32)
    displs = np.empty(numproc, dtype=np.int32)

    rcounts[0] = ave if res == 0 else ave + 1
    displs[0] = 0
    for k in range(1, numproc):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave 
        displs[k] = displs[k - 1] + rcounts[k - 1] 
else:
    size = np.empty(1, dtype=np.int32)
    rcounts = np.empty(numproc, dtype=np.int32)
    displs = np.empty(numproc, dtype=np.int32)

comm.Bcast([size, 1, MPI.INT], root=0)
comm.Bcast([rcounts, numproc, MPI.INT], root=0)
comm.Bcast([displs, numproc, MPI.INT], root=0)

if rank == 0:
    with open("x.txt", 'r', encoding = 'utf-8') as f:
        x_part = np.empty((size[0], rcounts[0]), dtype=np.float64)
        for i in range(rcounts[0]):
            for j in range(size[0]):
                x_part[j, i] = float(f.readline())

        for k in range(1, numproc):
            x_part_ = np.empty((size[0], rcounts[k]), dtype=np.float64)
            for i in range(rcounts[k]):
                for j in range(size[0]):
                    x_part_[j, i] = np.float64(f.readline())
            comm.Send([x_part_, rcounts[k] * size[0], MPI.DOUBLE], dest=k)
else:
    x_part = np.empty((size[0], rcounts[rank]), dtype=np.float64)
    comm.Recv([x_part, size[0] * rcounts[rank], MPI.DOUBLE], source=0)

if rank == 0:
    with open("y.txt", 'r', encoding = 'utf-8') as f:
        y = np.empty(size[0], dtype=np.float64)
        for i in range(size[0]):
            y[i] = np.float64(f.readline())
else:
    y = np.empty(size, dtype=np.float64)

comm.Bcast([y, size, MPI.DOUBLE], root=0)

lambda_ = 2e-4
s0 = 1
p = 0.5
w_part = np.zeros(rcounts[rank], dtype=np.float64)
iteration = 0

start_time = time.time()
while iteration < num_of_iters:
    lr = lambda_ * (s0 / (s0 + iteration)) ** p    
    w_part += - lr * (- 2 / size) * np.dot((y - np.dot(x_part, w_part)).T, x_part)
    iteration += 1

if rank == 0:
    w = np.empty(dimension[0])
else:
    w = None

comm.Gatherv([w_part, rcounts[rank], MPI.DOUBLE], [w, rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    euclidean_metric = np.sqrt(np.sum((w - hyperplane_parametrs) ** 2))
    time_in_secs = time.time() - start_time

    print('Original hyperplane:', hyperplane_parametrs)
    print('Estimation of hyperplane parameters:', w)
    print('Euclidean metric between original hyperplane and estimation:', euclidean_metric)
    print('Computation time in seconds:', time_in_secs)
    
