# mnist network 1.0
 run this code
 import mnist_data as mnist
 import network as nn
 net = nn.Network([784, 10], fill = 'random_sample', a = -1, b = 1)
 net.stochastic_gradient_descent(mnist.data[0:100], mnist.labels[0:100], 1.2, 10, 100, mnist.data_test, mnist.labels_test)
