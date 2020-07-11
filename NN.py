#The defaulty neural network with defaulty parameters to be evolved using NEAT. 
import numpy as np


initial_money = 10000
window_size = 30
skip = 1

class neuralnetwork:
    def __init__(self, id_, hidden_size = 128):
        self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
        self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
        self.fitness = 0
        self.id = id_

def relu(X):
    return np.maximum(X, 0)
    
def softmax(scores):
  return np.exp(scores)/sum(np.exp(scores), axis=0)

def feed_forward(X, nets):
    layer1 = np.dot(X, nets.W1)
    layer2 = relu(layer1)
    layer3 = np.dot(layer2, nets.W2)
    return softmax(a2)
