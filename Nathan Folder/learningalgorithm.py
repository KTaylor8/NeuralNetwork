import time
import numpy as np
import math
import os
import pandas
import random
import csv
miniBatch = np.random.randint(0, high=1, size=[3, 0])
numLayers = len(miniBatch)
w = np.random.randint(0, 1, size=[3, 3])
b = np.random.randint(0, 1, size=[3, 3])
miniBatch = np.random.randint(0, high=1, size=[3, 0])
numLayers = len(miniBatch)
inputs = np.random.randint(0, 1, size=[3, 3])
output = np.random.randint(0, 1, [1, 3])
zList = np.dot(inputs, w)
z = 0
def feedforward(self, z):
    z = np.dot(inputs, b)
    return z

def sigmoid(self, z):
    """Function for the activation function. Used to calculate the output of each neuron and the derivative of itself"""
    s = (1/(1 + math.exp(z*(-1))))
    return(s)

def sigmoidprime(self, z):
    """Function for the derivative of the activation function. Used to find the error of each neuron"""
    sp = (sigmoid(z) * (1 - sigmoid(z)) for z in zList)
    return sp

def costderivative(self, output, y):
    """Function for the derivative of the cost function. Used to find the error of each neuron"""
    return (output - y)

class backpropagation():
    w = np.random.randint(0, 1, size=(3, 3))
    b = np.random.randint(0, 1, size=(3, 3))
    
    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData
    = None):
        n = len(trainingData)
        for k in range(epochs):
            miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
        for miniBatch in miniBatches:
            self.updateWB(miniBatch, learningRate, w, b)
            """Still working on how to end an epoch/exit out of program"""

        if testData:
            print ("Epoch Over")

    def backprop(self, nablaB, nablaW, numLayers):
            """This function calculates the rate of change of the cost function and the biases/weights, uses that to find the error of each neuron, and uses the error to calculate the change in weights and biases. Variables are self, the change in b and a, and the number of layers in a minibatch"""
            #define activation, weighted inputs, and set up all lists/tuples
            activation = x
            activations = [x]
            zList = []

            """Feedforward section of the network. Calculates the activation for each neuron of the network and """
            z = (np.dot(self.w, input) + b)
            zList.append(z)
            activations = (self.sigmoid(x) for x in zList)

            errorL = self.costderivative((activations[numLayers], y) for y in output) * self.sigmoidprime(z[numLayers])

            nablaB[numLayers] = errorL
            nablaW[numLayers] = np.dot(errorL, activations[numLayers - 1].transpose())

            for l in range(2, self.numLayers):
                z = zList[numLayers]
                sp = self.sigmoidprime(z)
                errorL = np.dot(self.w[-l+1].transpose(), errorL) * sp
                nablaB = errorL
                nablaW = np.dot(errorL, activations[-l-1].transpose())

            return nablaB, nablaW

    def updateWB(self, miniBatch, learningRate, w, b):
        """Updates the weights and biases of the network based on the partial derivatives of the cost function. Variables are self (class specific variable), the list miniBatch, and the learning rate"""
    
        nablaW = np.zeros((w.shape) for x in self.w)
        nablaB = np.zeros((b.shape) for x in self.b)

        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(self, x, y, numLayers)

            nablaW = (nablaW + deltaNablaW for nablaW, deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB, deltaNablaB in zip(nablaB, deltaNablaB))

            self.w = (w - (learningRate/len(miniBatch))*nablaW for w, nablaW in zip(w, nablaW))

            self.b = (b - (learningRate/len(miniBatch))*nablaB for b, nablaB in zip(b, nablaB))

def main():
    minibatch = np.array([i] for i in input("Enter number"))
    w = np.random.rand(0, 1, [3, 3])
    b = np.random.rand(0, 1, [3, 3])
