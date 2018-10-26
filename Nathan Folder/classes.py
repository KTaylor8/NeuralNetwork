import numpy as np
import math
import time
import random
import os

#create an object for the neural network
class network(object):
    
    #creates weights and biases for each neuron for each layer
    def __init__(self, sizes):
        #sizes is the number of neurons in each layer
        self.num_layers = len(sizes)
        #uses gaussian distribution, with mean of 0 and standard deviation of 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def sigmoid(self, x)


