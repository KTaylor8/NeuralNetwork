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
inputs = np.random.randint(0, 1, size=[3, 3])
output = np.random.randint(0, 1, [1, 3])
zList = np.dot(inputs, w)
x = 0

def getMiniBatch():
    """Reads the ticTacToeData csv file, finds a line of code, and enters that into a list to create the first miniBatch. It also replaces each x, o, or b with either a one or a zero, and creates a list of inputs and outputs to be used in the rest of the network"""
    with open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\Nathan Folder\ticTacToeData.csv", "r", newline='') as dataFile:
        trainingData = tuple(dataFile)

        miniB = dataFile.readline()
        print({miniB})
        time.sleep(2)

        miniBSplit = miniB.split(",")
        print(miniBSplit)
        time.sleep(2)

        miniBatchInputs = miniBSplit[0:8]
        print(miniBatchInputs)

        for i in range(len(miniBatchInputs)):
            if miniBatchInputs[i] == 'x':
                miniBatchInputs[i] = 1.0
            else:
                miniBatchInputs[i] = 0.0

        miniBatchInputs = tuple(miniBatchInputs)
        print(f'miniBatchInputs = {miniBatchInputs}')

        output = miniBSplit[9]
        print(output)
        time.sleep(2)
        print(f'Minibatch is {miniBatchInputs}')
    return output, miniBatchInputs, trainingData

class backpropagation():
    
    def __init__(self, inputs, learningRate, numLayers, w, b):
        self.inputs = inputs
        self.numLayers = int(len(inputs))
        self.w = w
        print(self.w)
        time.sleep(2)

        self.trainingData = trainingData
        self.b = b
        print(self.b)
        time.sleep(2)

        self.learningRate = input("Learning rate = ")
        
    def feedforward(self, z):
        """Feedforward part: finding weighted sum of the weights and inputs, then adds bias"""
        z = np.add(np.dot(inputs, w), b)
        return z

    def sigmoid(self, z):
        """Function for the activation function. Used to calculate the output of each neuron and the derivative of itself"""
        s = (1/(1 + math.exp(z*(-1))))
        return(s)

    def sigmoidprime(self, s):
        """Function for the derivative of the activation function. Used to find the error of each neuron"""
        sp = (self.sigmoid(s) - self.sigmoid(s)**2)
        return sp

    def costderivative(self, s, y):
        """Function for the derivative of the cost function. Used to find the error of each neuron"""
        return (s - y)

    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData
    = None):
        """This part of the program will 
        create the miniBatch from the epoch
        run the gradient descent on that miniBatch
        repeat for remaining epoch
        switch to next epoch
        End program when test data runs out"""
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

            errorL = self.costderivative((activations[numLayers], output)) * self.sigmoidprime(z[numLayers])

            nablaB[numLayers] = errorL
            nablaW[numLayers] = np.dot(errorL, activations[numLayers - 1].transpose())

            for l in range(2, numLayers):
                z = zList[numLayers]
                sp = self.sigmoidprime(z)
                errorL = np.dot(self.w[-l+1].transpose(), errorL) * sp
                nablaB = errorL
                nablaW = np.dot(errorL, activations[-l-1].transpose())

            return nablaB, nablaW

    def updateWB(self, inputs, learningRate, w, b):
        """Updates the weights and biases of the network based on the partial derivatives of the cost function. Variables are self (class specific variable), the list miniBatch, and the learning rate"""
    
        nablaW = np.zeros(w.shape)
        nablaB = np.zeros(b.shape)

        for [input, output] in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(self, nablaB, nablaW, numLayers)

            nablaW = (nablaW + deltaNablaW for nablaW, deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB, deltaNablaB in zip(nablaB, deltaNablaB))

            self.w = (w - (learningRate/len(miniBatch))*nablaW for w, nablaW in zip(w, nablaW))

            self.b = (b - (learningRate/len(miniBatch))*nablaB for b, nablaB in zip(b, nablaB))



def main():
    #transpose miniBatch for input layer
    output, miniBatchInputs, trainingData = getMiniBatch()

    #Debugging
    print(output)
    print(miniBatchInputs)
    inputs = miniBatchInputs
    inputs = np.transpose(inputs)
    print(inputs)
    time.sleep(2)

    numLayers = int(len(inputs))

    w = [np.random.rand(numLayers,3)]
    print(w)
    time.sleep(2)

    b = [np.random.rand(numLayers, 3)]
    print(b)
    time.sleep(2)

    learningRate = input("Learning rate = ")
    #self is not within class
    backprop1 = backpropagation(inputs, learningRate, numLayers, w, b)
    backprop1.updateWB(inputs, learningRate, w, b)

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()