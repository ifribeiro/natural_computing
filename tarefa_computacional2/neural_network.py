import numpy as np
from scipy.special import softmax

def mse(y, y_hat):
    return np.sum((y_hat - y)**2)/y.size

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax_(x):
    return softmax(x.T, axis=1)

def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

def ReLU(x):
    return x*(x>0)

def CategoricalCrossentropy(y_hat, y_true):
        
    m = y_true.shape[1]

    epsilon = 1e-07
    # y_hat and y_true (1, m)
    cce = np.sum(y_true * np.log(y_hat + epsilon), axis=0)

    cce = (-1.0 / m) * np.sum(cce)
    return cce

# Loss Functions 
def logloss(y, a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))

def d_logloss(y, a):
    return (a - y)/(a*(1 - a))

# The layer class
class Layer:

    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'softmax':(softmax_, d_sigmoid),
        'relu':(ReLU,d_sigmoid)
    }
    learning_rate = 0.1

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs)
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunctions.get(activation)
        self.shpW = self.W.shape
        self.shpb = self.b.shape
        

    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z).astype('float')
        return self.A

    def backprop(self, dA):
        dZ = np.multiply(self.d_act(self.Z), dA)
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return dA_prev