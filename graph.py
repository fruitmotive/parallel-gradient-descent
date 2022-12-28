from matplotlib import pyplot as plt

# Первый график
par_1 = [12.532922267913818, 14.483150005340576, 16.781116724014282, 14.612939834594727]
seq = [12.209344387054443, 12.209344387054443, 12.209344387054443, 12.209344387054443]

params = {'size': 100000, 'dimension': 4, 'max_iter': 10000}

plt.figure(figsize=(12, 6))
plt.plot(list(range(1, 5)), par_1, label='Параллельный алгоритм')
plt.plot(list(range(1, 5)), seq, label='Последовательный алгоритм')
plt.xlabel('Кол-во процессов')
plt.ylabel('Время вычислений в секундах')
plt.legend()
plt.title('Размер выборки: {0}, Размерность данных: {1}, Кол-во итераций: {2}'.format(params['size'], params['dimension'], params['max_iter']))
plt.savefig('graph_1.png')
plt.show()


