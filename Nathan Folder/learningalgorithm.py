import time
import numpy as np
import math
import os
import pandas

class backpropagation:
    
    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData = None):
        n = len(trainingData)
        for x in xrange(epochs):
            miniBatches = [trainingData[k:k+miniBatchSize] for k in xrange(0, n, miniBatchSize)]
        for miniBatch in miniBatches:
            self.updateWB(miniBatch, learningRate)
            
    def updateWB(self, miniBatch, learningRate):

        nablaW = np.zeros((w.shape) for x in self.w)
        nablaB = np.zeros((b.shape) for x in self.b)

        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = backprop(x, y)

            nablaW = (nablaW + deltaNablaW for nablaW, deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB, deltaNablaB in zip(nablaB, deltaNablaB))

            self.w = (w - (learningRate/len(miniBatch))*nablaW for w, nablaW in zip(w, nablaW))

            self.b = (b - (learningRate/len(minibatch))*nablaB for b, nablaB in zip(b, nablaB))

    def backprop(self, nablaB, nablaW, numlayers):

        #define activation, weighted inputs, and set up all lists/tuples
        activation = x
        activations = [x]
        zList = []

        #feedforward
        z = (np.dot(self.w, input) + b)
        zList.append(z)
        activations = (sigmoid(x) for x in zList)

        # activation = x
        """activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #error in output layer
        errorO = np.dot(nablaA, sigmoidprime(l))"""

        errorL = self.costderivative(activations[numlayers], y) * sigmoidprime(z[numlayers])

        nablaB[numlayers] = errorL
        nablaW[numlayers] = np.dot(errorL, activations[numlayers - 1].transpose())

        for l in xrange(2, self.numlayers):
            z = zList[numlayers]
            sp = sigmoidprime(z)
            errorL = np.dot(self.w[-l+1].transpose(), delta) * sp
            nablaB = delta
            nablaW = np.dot(delta, activations[-l-1].transpose())
        """# backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)"""

        return NablaB, NablaW

    def sigmoidprime(z):
        sp = sigmoid(z) * (1 - sigmoid(z))
        return sp

    def costderivative(self, output, y):
        return (output - y)