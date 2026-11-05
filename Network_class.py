# Michael Nielsen book explanation

import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)

class NN(object):

    def __init__(self, nn_structure):
        self.NN_structure_len = len(nn_structure)
        self.w = [np.random.randn(b, a) for a, b in zip(nn_structure[:-1], nn_structure[1:])]
        self.b = [np.random.randn(b, 1) for b in nn_structure[1:]]


    def sigma(self, x):
        return 1 / (1 + np.exp(-x))


    def sigma_derivative(self, x):
        s = self.sigma(x)
        return s * (1 - s)


    def feedforward(self, a):

        # We need to create a system equation consist of weight and biases for each layer (hidden and outer). Therefore in first case we make forward way.
        for temp_w, temp_b in zip(self.w, self.b):
            z = temp_w @ a + temp_b
            a = self.sigma(z)

        return a
        

    def SGD(self, training_data, epochs, mini_batch_len, eta, test_data = None):
        
        for epoch in range(0, epochs):
            random.shuffle(training_data)

            for i in range(0,len(training_data)//mini_batch_len):
                mini_batch = training_data[i*mini_batch_len: (i+1)*mini_batch_len]
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data)
                logging.info(f"For epoch {epoch} accuracy is {accuracy}")
            

    def evaluate(self, test_data):

        trues_count = 0
        for x, y in test_data:
            if np.argmax(self.feedforward(x)) == y:
                trues_count+=1
        
        return trues_count/len(test_data)

        
    def backprop(self, x, y):
        
        n = self.NN_structure_len
        a, z = [], []     
        a.append(x)
        nabla_C0_w, nabla_C0_b = [], []
        
        # делаем прямой проход, чтобы заполнить системы уранений для скрытых и внешнего слоёв
        for i in range(0, len(self.w)):
            z.append( self.w[i] @ a[i] + self.b[i] )
            a.append( self.sigma(z[i]) )

        # заполняем аккумуляторы для списка из кортежей ошибок
        for w, b in zip(self.w, self.b):
            nabla_C0_w.append(np.zeros(w.shape))
            nabla_C0_b.append(np.zeros(b.shape))
            
        # разовая формула ошибки (только для выходного слоя)
        delta_L = 2 * (a[-1] - y) * self.sigma_derivative(z[-1]) # BP1
        
        # в аккумуляторах[1] заполняем ошибки для выходного слоя по разовой формуле
        nabla_C0_w[-1] = delta_L @ a[-2].transpose()
        nabla_C0_b[-1] = delta_L

        # в аккумуляторах[0] заполняем уже всё по рекурсивным формулам.
        for i in range(0, n-2):
            delta_l = ( self.w[-1-i].transpose() @ delta_L ) * self.sigma_derivative(z[-2-i]) # BP2
            delta_L = delta_l
            nabla_C0_w[n-3-i] = delta_l @ a[n-3-i].transpose() # BP4
            nabla_C0_b[n-3-i] = delta_l # BP3
        
        return (nabla_C0_w, nabla_C0_b)


    def update_mini_batch(self, mini_batch, eta):

        nabla_C0_w, nabla_C0_b = [], []
        
        # вновь заполняем аккумуляторы для набл весов [матр, матр] и смещений [столбец, столбец]
        for w, b in zip(self.w, self.b):
            nabla_C0_w.append(np.zeros(w.shape))
            nabla_C0_b.append(np.zeros(b.shape))

        # вытаскиваем из пачки входных данных n раз столбцы x и y
        # закидываем их в бэкпроп, получаем временные значения набл
        # суммируем n раз наблы, обновляя их раз за разом
        n = len(mini_batch)
        for i in range(0, n):
            x, y = mini_batch[i][0], mini_batch[i][1]
            temp_nabla_C0_w, temp_nabla_C0_b = self.backprop(x, y)
            
            for j in range(0, len(self.w)):
                nabla_C0_w[j] += temp_nabla_C0_w[j]
                nabla_C0_b[j] += temp_nabla_C0_b[j]

        # рассчитывем веса согласно усредненному значению набл
        for j in range(0, len(self.w)): 
            self.w[j] = self.w[j] - eta*nabla_C0_w[j]/n
            self.b[j] = self.b[j] - eta*nabla_C0_b[j]/n

    
    def print_matrix(self, data, name="Объект"):
        if isinstance(data, (list, tuple)):
            for i, item in enumerate(data, start=1):
                self.print_matrix(item, name=f"{name} [{i}]")
            return
        matrix = np.asarray(data)
        print(np.array2string(matrix, precision=3, suppress_small=True))
        print()

