import numpy as np

# activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    y=np.where(x>0,1,0)
    return y

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def cross_entropy_prime(y, y_hat):
    return -np.divide(y, y_hat)
