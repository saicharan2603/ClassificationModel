import numpy as np
import matplotlib.pyplot as plt
from mlp_model import FCLayer, ActivationLayer, Network
from activation_error_functions import relu, relu_prime, softmax, softmax_prime, cross_entropy, cross_entropy_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(512, 64))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(64, 64))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(64, 10))
net.add(ActivationLayer(softmax, softmax_prime))

# train
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train, y_train, epochs=10, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)



