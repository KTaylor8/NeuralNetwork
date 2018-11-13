import time
import numpy as np
import math
import os
import pandas

class backpropagation:

    def update_w_b(self, ):

        nabla_w = np.zeros((w.shape) for x in self.weights)
        nabla_b = np.zeros((b.shape) for x in self.biases)

        delta_nabla_w, delta_nabla_b = backprop(self.weights, self.biases)

        nabla_w = (nabla_w + delta_nabla_w for nabla_w, delta_nabla_w in (x,y))
        nabla_b = (nabla_b + delta_nabla_b for nabla_b, delta_nabla_b in (x,y))

        self.weights = (self.weights + nabla_w)
        self.biases = (self.biases + nabla_b)

       


    def backprop(self, nabla_b, nabla_w):

        #define activation, weighted inputs, and set up all lists/tuples
        activation = x
        activations = [x]
        zs = []

        #feedforward
        z = (np.multiply(w, inp) + b)
        act = sigmoid(zs)

        #error in output layer
        error_o = nabla_a * sigmoidprime(l)

        #backpropagation
        error_l = (np.transpose(self.weights(l)*sigmoidprime(l)))

        #find roc for biases + apply
        delta_nabla_b = error_l

        #find roc for weights + apply
        delta_nabla_w = error_l * self.activations()