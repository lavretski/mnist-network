import numpy as np
import random

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_sigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2

def cost_derivative(a, label):
    return (a - y(label))

def y(x):
    res = np.zeros(10)
    res[x] = 1
    res = res[:, np.newaxis]
    return res
    

    

class Network:
    
    def __init__(self, layers, fill = 'random_sample', a = 0, b = 0):
        if layers[0] != 784 or layers[-1] != 10:
            raise ValueError ('is this a picture network?')
        self.len_layers = len(layers)
        self.layers = layers
        if self.len_layers < 2:
            raise ValueError ('n < 2')
        if fill == 'ones':
            self.weights = [np.ones((layers[i+1], layers[i])) for i in range(self.len_layers - 1)]#крайній випадок коли масив всередині якого один масив
            self.biases = [np.ones((layers[i+1], 1)) for i in range(self.len_layers - 1)]# тут можуть бути якість проблеми що у мене list з np.arrays
        elif fill =='random_sample':
            self.weights = [a + (b - a) * np.random.random_sample((layers[i+1], layers[i])) for i in range(self.len_layers - 1)]
            self.biases = [a + (b - a) * np.random.random_sample((layers[i+1], 1)) for i in range(self.len_layers - 1)]
        else: raise ValueError ('unknown fill')
        
        
        
    def feedforward(self, row_input):
        picture = row_input.flatten().reshape((784,1))
        res = picture
        layers_output = [picture]     
        sigmoid_derivatives = [] 
        for i in range(self.len_layers - 1):
            z = self.weights[i] @ res + self.biases[i]
            res = sigmoid(z)
            layers_output.append(res)
            sigmoid_derivatives.append(derivative_sigmoid(z))
        return layers_output, sigmoid_derivatives
    
    
    def index_yield(self, data_len, m): 
        for i in range(0, data_len, m):
            if i == data_len:
                pass
            yield (i, i+m)
    
    def stochastic_gradient_descent(self, data, labels, epoch, learning_rate, m, data_test = [], labels_test = []):
        for i in range(epoch):
            items = list(zip(data, labels))
            random.shuffle(items)
            data = [item[0] for item in items]
            labels = [item[1] for item in items] #працює але не панятно, чи це діюче, пробував з і без, зіба що стартує повільніше, але потім +- таксамо взлітає
            for k, j in self.index_yield(len(data), m):
                
                delta_b_raw, delta_w_raw = self.backpropagation(data[k:j], labels[k:j])#мені прийшли ваги з одинаковими числами що таке
                
                delta_b = [array/(j-k) *(-learning_rate) for array in delta_b_raw]#як мінімум вони того що треба розміру
                delta_w = [array/(j-k) *(-learning_rate) for array in delta_w_raw]
                
                self.weights = [a + b for a, b in zip(self.weights, delta_w)]
                self.biases = [a + b for a, b in zip(self.biases, delta_b)]
            if len(data_test) and len(labels_test):
                print(self.test_network(data_test, labels_test), 'score')
            print(f'end of the {i} epoch')

    def backpropagation(self, batch, labels):
        weights_derv_batch_sum = [np.zeros_like(w) for w in self.weights]
        biases_derv_batch_sum = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(batch)):
            layers_output, sigmoid_derivatives = self.feedforward(batch[i])
            cost_der = cost_derivative(layers_output[-1], labels[i])
            
            error = cost_der * sigmoid_derivatives[-1]
            biases_derv_batch_sum[-1] = biases_derv_batch_sum[-1] + error
            weights_derv_batch_sum[-1] = weights_derv_batch_sum[-1] + layers_output[-2].ravel() * error#потрібно картинку теж враховувати
            for j in range(self.len_layers - 2, 0, -1):#закнічуємо на післяпершому слої
                error = self.weights[j].T @ error * sigmoid_derivatives[j-1] # 0 ваги не юзаються, бо немає ошибки в першого слою
                biases_derv_batch_sum[j - 1] = biases_derv_batch_sum[j - 1] + error
                weights_derv_batch_sum[j - 1] = weights_derv_batch_sum[j - 1] + layers_output[j - 1].ravel() * error
        return biases_derv_batch_sum, weights_derv_batch_sum
            
    def test_network(self, test_data, labels):
        count = 0
        for i in range(len(test_data)):
            picture = test_data[i].squeeze().reshape((784,1))
            res = picture
            for j in range(self.len_layers - 1):
                z = self.weights[j] @ res + self.biases[j]
                res = sigmoid(z)
            if res.argmax() == labels[i]:
                count += 1
        return count