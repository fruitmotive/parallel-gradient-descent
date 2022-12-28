import numpy as np
from threadpoolctl import threadpool_limits
import time 

start_time = time.time()

num_of_iters = 100000

# Считывание данных из файла 
with open("parameters.txt", 'r', encoding = 'utf-8') as f:
    dimension = int(f.readline())
    size = int(f.readline())
    hyperplane_parametrs = np.array([int(i) for i in f.readline().split()])

with open("x.txt", 'r', encoding = 'utf-8') as f: 
    x = np.empty((size, dimension))
    for i in range(dimension):
        for j in range(size):
            x[j, i] = float(f.readline())

with open("y.txt", 'r', encoding = 'utf-8') as f:
    y = np.empty(size)
    for i in range(size):
        y[i] = float(f.readline())

# Начальное состояние модели 
lambda_ = 2e-4
s0 = 1
p = 0.5
iteration = 0
w = np.zeros(dimension)

# Процесс обучения
while iteration < num_of_iters:
    lr = lambda_ * (s0 / (s0 + iteration)) ** p    
    gradient = (- 2 / size) * np.dot((y - np.dot(x, w)).T, x)
    returning = - lr * gradient
    w += returning
    iteration += 1

euclidean_metric = np.sqrt(np.sum((w - hyperplane_parametrs) ** 2))

time_in_secs = time.time() - start_time

print('Original hyperplane:', hyperplane_parametrs)
print('Estimation of hyperplane parameters:', w)
print('Euclidean metric between original hyperplane and estimation:', euclidean_metric)
print('Computation time in seconds:', time_in_secs)

