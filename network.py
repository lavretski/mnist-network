import numpy as np
import random

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_sigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2

def y1(x):
    res = np.zeros(10)
    res[x] = 1
    res = res[:, np.newaxis]
    return res

def y2(x):
    res = np.zeros(10)
    res[x] = 1
    #res = res[:, np.newaxis]
    return res
    

    

class Network:
    
    def __init__(self, layers, fill = 'random', a = 0, b = 0):
        #треба буде добавити сюда докстрінги в кожену з функцій і в сам клас
        if layers[0] != 784 or layers[-1] != 10:
            raise ValueError ('is this a picture network?')
        self.len_layers = len(layers)
        self.layers = layers
        if self.len_layers < 2:
            raise ValueError ('n < 2')
        if fill == 'ones':
            self.weights = [np.ones((layers[i+1], layers[i])) for i in range(self.len_layers - 1)]
            self.biases = [np.ones((layers[i+1], 1)) for i in range(self.len_layers - 1)]
        elif fill =='random_sample':
            self.weights = [a + (b - a) * np.random.random_sample((layers[i+1], layers[i])) for i in range(self.len_layers - 1)]
            self.biases = [a + (b - a) * np.random.random_sample((layers[i+1], 1)) for i in range(self.len_layers - 1)]
        elif fill =='random':
            self.weights = [np.random.randn(layers[i+1], layers[i]) for i in range(self.len_layers - 1)]
            self.biases = [np.random.randn(layers[i+1], 1) for i in range(self.len_layers - 1)]
        else: raise ValueError ('unknown fill')
        
        
    def feedforward1(self, row_input):
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
    
    def feedforward2(self, batch_raw):
        batch = batch_raw.reshape(len(batch_raw), 784)
        res = batch
        layers_output_batch = [batch, ]     
        sigmoid_derivatives_batch = [] 
        for i in range(self.len_layers - 1):
            z = (self.weights[i] @ res.T + self.biases[i]).T
            res = sigmoid(z)
            layers_output_batch.append(res)
            sigmoid_derivatives_batch.append(derivative_sigmoid(z))
        return layers_output_batch, sigmoid_derivatives_batch
    
    
    def index_yield(self, data_len, m): 
        for i in range(0, data_len, m):
            yield (i, i+m)
    
    def SGD(self, data, labels, epoch, learning_rate, m, data_test = [], labels_test = [],  BP = 'bp1'):
        for i in range(epoch):
            #items = list(zip(data, labels))
            #random.shuffle(items)#тут вихідний тип не підходить
            #data = [item[0] for item in items]
            #labels = [item[1] for item in items] #працює але не панятно, чи це діюче, пробував з і без, зіба що стартує повільніше, але потім +- таксамо взлітає
            for k, j in self.index_yield(len(data), m):
                if BP == 'bp1':
                    delta_b_raw, delta_w_raw = self.backpropagation1(data[k:j], labels[k:j])
                if BP == 'bp2':
                    delta_b_raw, delta_w_raw = self.backpropagation2(data[k:j], labels[k:j])
                self.weights = [a + b/(j-k) *(-learning_rate) for a, b in zip(self.weights, delta_w_raw)]
                self.biases = [a + b/(j-k) *(-learning_rate) for a, b in zip(self.biases, delta_b_raw)]
                
            if len(data_test) and len(labels_test):
                print(self.test_network(data_test, labels_test), 'score')
            print(f'end of the {i} epoch')
    
    def backpropagation1(self, batch, labels):
        weights_derv_batch_sum = [np.zeros_like(w) for w in self.weights]
        biases_derv_batch_sum = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(batch)):
            layers_output, sigmoid_derivatives = self.feedforward1(batch[i])
            cost_der = layers_output[-1] - y1(labels[i])
            
            error = cost_der * sigmoid_derivatives[-1]
            biases_derv_batch_sum[-1] = biases_derv_batch_sum[-1] + error
            weights_derv_batch_sum[-1] = weights_derv_batch_sum[-1] + layers_output[-2].ravel() * error
            for j in range(self.len_layers - 2, 0, -1):
                error = self.weights[j].T @ error * sigmoid_derivatives[j-1]
                biases_derv_batch_sum[j - 1] = biases_derv_batch_sum[j - 1] + error
                weights_derv_batch_sum[j - 1] = weights_derv_batch_sum[j - 1] + layers_output[j - 1].ravel() * error
        return biases_derv_batch_sum, weights_derv_batch_sum
            
    def backpropagation2(self, batch, labels):
        m = len(batch)
        weights_derv_batch_sum = [np.zeros_like(w) for w in self.weights]
        biases_derv_batch_sum = [np.zeros_like(b) for b in self.biases]
        
        labels_vectors = np.array([y2(labels[i]) for i in range(m)])
        
        layers_output_batch, sigmoid_derivatives_batch = self.feedforward2(batch)
        
        cost_der_batch = layers_output_batch[-1] - labels_vectors
        error = cost_der_batch * sigmoid_derivatives_batch[-1]
        
        biases_derv_batch_sum[-1] = error.sum(axis = 0)[..., np.newaxis]
        weights_derv_batch_sum[-1] = (error[..., np.newaxis] * layers_output_batch[-2][:, np.newaxis, :]).sum(axis = 0)
        
        for j in range(self.len_layers - 2, 0, -1):
            error = (self.weights[j].T @ error.T).T * sigmoid_derivatives_batch[j-1]
            biases_derv_batch_sum[j - 1] = biases_derv_batch_sum[j - 1] + error.sum(axis = 0)[..., np.newaxis]
            weights_derv_batch_sum[j - 1] = (error[..., np.newaxis] * layers_output_batch[j - 1][:, np.newaxis, :]).sum(axis = 0)#оця операція хаває багато часу, коли багато обчислень ff вже перестаєж навіть помагати і 1.0 
        
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
