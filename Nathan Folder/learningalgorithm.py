import time
import numpy as np
import math
import os
import pandas

class backpropagation:
    import class network
    

    def update_w_b(self, minibatch, l_r):

        nabla_w = np.zeros((w.shape) for x in self.weights)
        nabla_b = np.zeros((b.shape) for x in self.biases)

        for x, y in minibatch:
            delta_nabla_b, delta_nabla_w = backprop(x, y)

            nabla_w = (nabla_w + delta_nabla_w for nabla_w, delta_nabla_w in zip(nabla_w, delta_nabla_w))

            nabla_b = (nabla_b + delta_nabla_b for nabla_b, delta_nabla_b in zip(nabla_b, delta_nabla_b))

            self.weights = (weights - (l_r/len(minibatch))*nabla_w for weights, nabla_w in zip(weights, nabla_w))

            self.biases = (biases - (l_r/len(minibatch))*nabla_b for weights, nabla_b in zip(weights, nabla_b))

    def backprop(self, nabla_b, nabla_w):

        #define activation, weighted inputs, and set up all lists/tuples
        activation = x
        activations = [x]
        zs = []

        #feedforward
        zlist = (np.multiply(w, inp) + b)
        act = sigmoid(zs)

        #error in output layer
        error_o = np.dot(nabla_a, sigmoidprime(l))

        #error in other layers
        for l in numlayers:
            lweight = np.transpose(weights)
            error_i = np.dot(((x*e), sigmoidprime(z) for x, e, z in lweight, error_o, zlist))

        #find roc for biases + apply
        delta_nabla_b = error_l

        #find roc for weights + apply
        delta_nabla_w = error_l * self.activations()

        return delta_nabla_b, delta_nabla_w