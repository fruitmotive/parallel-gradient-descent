from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numproc = comm.Get_size()

def generate_sample(numproc, dimension, size):
    a = -10 
    b = 10
    noise_sigma = 1
    real_w = np.ones(dimension)
    x = (b - a) * np.random.rand(size, dimension) + a
    y = x @ real_w + np.random.normal(scale=noise_sigma, size=size)

    rcounts = np.empty(numproc, dtype=int)
    displs = np.empty(numproc, dtype=int)
    ave, res = np.divmod(size, numproc)

    rcounts[0] = ave if res == 0 else ave + 1
    displs[0] = 0

    for k in range(1, numproc):
        if k < 1 + res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave 
        displs[k] = displs[k - 1] + rcounts[k - 1]

    x_returning, y_returning = [], []
    for k in range(numproc):
        x_returning.append(x[displs[k]:displs[k] + rcounts[k], :])
        y_returning.append(y[displs[k]:displs[k] + rcounts[k]])
    
    return x_returning, y_returning

size = 100000
dimension = 4
max_iter = 10000

if rank == 0:
    x, y = generate_sample(numproc, dimension, size)
else:
    x, y = None, None

x_part = comm.scatter(x, root=0)
y_part = comm.scatter(y, root=0)

# Начальное состояние модели 
result = {}
lambda_ = 2e-4
s0 = 1
p = 0.5
w = np.zeros(dimension)
iteration = 0
loss_start_part = ((y_part - x_part @ w) ** 2).sum()
loss_start = comm.allreduce(loss_start_part) / size
result['loss_start'] = loss_start

# Процесс обучения
start_time = time.time()
while iteration <= max_iter:
    gradient_part = (- 2 / size) * (y_part - x_part @ w).T @ x_part
    gradient = comm.allreduce(gradient_part)
    lr = lambda_ * (s0 / (s0 + iteration)) ** p    
    returning = - lr * gradient
    w += returning
    iteration += 1
loss_fin_part = ((y_part - x_part @ w) ** 2).sum()
loss_fin = comm.allreduce(loss_fin_part) / size
result['loss_fin'] = loss_fin
result['time_in_secs'] = time.time() - start_time

if rank == 0:
    print('time_in_secs:', result['time_in_secs'])
    print('loss_start:', result['loss_start'])
    print('loss_fin:', result['loss_fin'])

    

