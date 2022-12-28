from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numproc = comm.Get_size()

def generate_sample(dimension, size):
    a = -10
    b = 10
    noise_sigma = 1
    real_w = np.ones(dimension)
    x = (b - a) * np.random.rand(size, dimension) + a
    y = x @ real_w + np.random.normal(scale=noise_sigma, size=size)
    return x, y

def generate_parts(x, y, dimension, numproc):
    rcounts = np.empty(numproc, dtype=int)
    displs = np.empty(numproc, dtype=int)
    ave, res = np.divmod(dimension, numproc)
    rcounts[0] = ave if res == 0 else ave + 1
    displs[0] = 0
    for k in range(1, numproc):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave 
        displs[k] = displs[k - 1] + rcounts[k - 1]
    x_returning, y_returning = [], []
    for k in range(numproc):
        x_returning.append(x[:, displs[k]:displs[k] + rcounts[k]])
        y_returning.append(y)
    return x_returning, y_returning

def training(x, y, size, num_iters):
    dimension = x.shape[1]
    lambda_ = 2e-4
    s0 = 1
    p = 0.5
    w = np.zeros(dimension)
    iteration = 0
    start_time = time.time()
    while iteration < num_iters:
        lr = lambda_ * (s0 / (s0 + iteration)) ** p    
        w += - lr * (- 2 / size) * (y - x @ w).T @ x
        iteration += 1
    return {'w': w, 'time_in_secs': time.time() - start_time}

size = 10000
dimension = 4
max_iter = 100000

if rank == 0:
    result = {}
    x, y = generate_sample(dimension, size)
    w = np.zeros(dimension)
    result['loss_start'] = ((y - x @ w) ** 2).mean()
    x_aux, y_aux = generate_parts(x, y, dimension, numproc)
else:
    x_aux, y_aux = None, None

x_aux, y_aux = comm.scatter(x_aux, 0), comm.scatter(y_aux, 0)
aux = training(x_aux, y_aux, size, max_iter)
result_part = aux['w']

w_fin = comm.gather(result_part, root=0)
time_ = comm.allreduce(aux['time_in_secs'], op=MPI.MAX)
if rank == 0:
    result['w'] = np.hstack(np.array(w_fin, dtype=object))
    result['loss_fin'] = ((x @ result['w'] - y) ** 2).mean()
    result['time_in_secs'] = time_

    print('time_in_secs:', result['time_in_secs'])
    print('loss_start:', result['loss_start'])
    print('loss_fin:', result['loss_fin'])
   
