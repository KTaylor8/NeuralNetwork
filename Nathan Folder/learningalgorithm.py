import time
import numpy as np
import math
import os
import pandas

class backpropagation:
    
    def updateWB(self, miniBatch, learningRate):

        nablaW = np.zeros((weights.shape) for x in self.weights)
        nablaB = np.zeros((biases.shape) for x in self.biases)

        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = backprop(x, y)

            nablaW = (nablaW + deltaNablaW for nablaW, deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB, deltaNablaB in zip(nablaB, deltaNablaB))

            self.weights = (weights - (learningRate/len(miniBatch))*nablaW for weights, nablaW in zip(weights, nablaW))

            self.biases = (biases - (learningRate/len(minibatch))*nablaB for weights, nablaB in zip(weights, nablaB))

    def backprop(self, nablaB, nablaW):

        #define activation, weighted inputs, and set up all lists/tuples
        activation = x
        activations = [x]
        zList = []

        #feedforward
        zList = (np.multiply(weights, input) + biases)
        act = sigmoid(zList)

        #error in output layer
        errorO = np.dot(nablaA, sigmoidprime(l))

        #error in other layers
        for l in numlayers:
            lWeight = np.transpose(weights)
            errorI = (np.dot(x, sigmoidprime(z) for x in lWeight))
            
        #find roc for biases + apply
        deltaNablaB = errorL

        #find roc for weights + apply
        deltaNablaW = errorL * self.activations()

        return deltaNablaB, deltaNablaW