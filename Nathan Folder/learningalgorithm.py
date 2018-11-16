import time
import numpy as np
import math
import os
import pandas

class backpropagation:
    

    def update_w_b(self, minibatch, l_r):

        NablaW = np.zeros((w.shape) for x in self.weights)
        NablaB = np.zeros((b.shape) for x in self.biases)

        for x, y in minibatch:
            DeltaNablaB, DeltaNablaW = backprop(x, y)

            NablaW = (NablaW + DeltaNablaW for NablaW, DeltaNablaW in zip(NablaW, DeltaNablaW))

            NablaB = (NablaB + DeltaNablaB for NablaB, DeltaNablaB in zip(NablaB, DeltaNablaB))

            self.weights = (weights - (l_r/len(minibatch))*NablaW for weights, NablaW in zip(weights, NablaW))

            self.biases = (biases - (l_r/len(minibatch))*NablaB for weights, NablaB in zip(weights, NablaB))

    def backprop(self, NablaB, Nablaw):

        #define activation, weighted inputs, and set up all lists/tuples
        activation = x
        activations = [x]
        zs = []

        #feedforward
        zlist = (np.multiply(w, inp) + b)
        act = sigmoid(zs)

        #error in output layer
        ErrorO = np.dot(NablaA, sigmoidprime(l))

        #error in other layers
        for l in numlayers:
            LWeight = np.transpose(weights)
            ErrorI = (np.dot(x, sigmoidprime(z) for x in LWeight))
        #find roc for biases + apply
        DeltaNablaB = ErrorL

        #find roc for weights + apply
        DeltaNablaW = ErrorL * self.activations()

        return DeltaNablaB, DeltaNablaW