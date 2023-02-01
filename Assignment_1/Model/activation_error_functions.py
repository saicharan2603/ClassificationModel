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
    s = softmax(x).reshape((-1, 1))

    s_prime = - np.matmul(s, s.T)
    for i in range(len(x)):
        s_prime[i][i] = s[i] * (1 - s[i])

    return s_prime

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def cross_entropy_prime(y, y_hat):
    return -np.divide(y, y_hat)
