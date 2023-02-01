import numpy as np

class NeuralNeworks:
    def __init__(self, input_neurons, h1_neurons, h2_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.h1_neurons = h1_neurons
        self.h2_neurons = h2_neurons
        self.output_neurons = output_neurons

        self.W1 = np.random.randn(self.input_neurons, self.h1_neurons)
        self.W2 = np.random.randn(self.h1_neurons, self.h2_neurons)
        self.W3 = np.random.randn(self.h2_neurons, self.output_neurons)

        self.b1 = np.random.randn(self.h1_neurons)
        self.b2 = np.random.randn(self.h2_neurons)
        self.b3 = np.random.randn(self.output_neurons)
        
    def relu(x):
    return np.maximum(0, x)


    def relu_dash(x):
        y = np.where(x > 0, 1, 0)
        return y


    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)


    def softmax_dash(x):
        s = softmax(x).reshape((-1, 1))

        s_prime = - np.dot(s, s.T)
        for i in range(len(x)):
            s_prime[i][i] = s[i] * (1 - s[i])

        return s_prime


    def cross_entropy(y, y_hat):
        return -np.sum(y * np.log(y_hat))


    def cross_entropy_dash(y, y_hat):
        return -np.divide(y, y_hat)


    def farword_pass(self, X):
        self.z1= np.dot(self.W1,X)+self.b1
        self.v1= relu(self.z1)
        self.z2= np.dot(self.W2,self.v1)+self.b2
        self.v2= relu(self.z2)
        self.z3= np.dot(self.W3,self.v2)+self.b3
        y_hat  = softmax(self.z3)
        return y_hat
    
    def backprop(self, X, y, lr=0.001)
        y_hat= self.farword_pass(X)
        

        dE_dy_hat= self.cross_entropy_dash(y,y_hat)
        dE_dz3= np.dot(self.softmax_dash(self.z3), dE_dy_hat) # Also equal to dE_db3
        dE_dW3= np.dot(dE_z3,self.v2.T)
        dE_db3= dE_dz3
        
        dE_dv2= np.dot(self.W3.T,dE_dz3)
        dE_dz2= np.multiply(dE_dv2,relu_dash(self.z2)) #Also equal to dE_db2
        dE_dW2= np.dot(dE_dz2,self.v1.T)
        dE_db2= dE_dz2

        dE_dv1= np.dot(self.W2.T, dE_dz2)
        dE_dz1= np.multiply(dE_dv1,relu_dash(self.z1)) # Also equal to dE_db1
        dE_dW1= np.dot(dE_dz1,X.T)
        dE_db1= dE_dz1

        # dE_dX = np.dot(self.W1,dE_dz1)
        self.W1 -= lr*dE_dW1
        self.W2 -= lr*dE_dW2
        self.W3 -= lr*dE_dW3
        self.b1 -= lr*dE_db1
        self.b2 -= lr*dE_db2
        self.b3 -= lr*dE_db3

        Error = self.cross_entropy(y, self.farword_pass(X))/X.shape[1]
    return Error
