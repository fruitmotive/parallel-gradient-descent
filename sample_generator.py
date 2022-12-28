import numpy as np

dimension = int(input('Enter sample dimension: '))
size = int(input('Enter sample size: '))
hyperplane_parametrs = np.array([i for i in range(1, dimension + 1)])

a = -10
b = 10
x = (b - a) * np.random.rand(size, dimension) + a
y = x @ hyperplane_parametrs

with open("parameters.txt", 'w', encoding = 'utf-8') as f:
    f.write(str(dimension) + '\n')
    f.write(str(size) + '\n')
    for i in hyperplane_parametrs:
        f.write(str(i) + ' ')
    f.write('\n')

with open("x.txt", 'w', encoding = 'utf-8') as f:    
    for i in range(dimension):
        for j in range(size):
            f.write(str(x[j, i]) + '\n')

with open("y.txt", 'w', encoding = 'utf-8') as f:
    for i in y:
        f.write(str(i) + '\n')