import csv
import os
import random
import numpy as np
import math

# asks user to set number of layers and neurons and outputs (9,9,1)
# randomizes weights and biases
# x = 1 and b,o = 0
# setting minibatch (row) that runs through each time and changes weights and biases based on accuracy


class network():

    def __init__(self, layerSizes):
        """
        This runs automatically to initialize the attributes for an instance of a class when the instance is created. It takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the weights and biases.
        """
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        # lists of 9 weights and 9 biases for each neuron in hidden layer
        self.b = [random.uniform(0, 1) for i in range(layerSizes[1])]
        self.w = [random.uniform(0, 1) for i in range(layerSizes[1])]
        # self.b = [np.random.randn(y, 1) for y in layerSizes[1:]]
        # self.w = [np.random.randn(y, x)
        #           for x, y in zip(layerSizes[:-1], layerSizes[1:])]
        print(
            f"Number of layers = {self.numLayers}\n"
            f"Sizes of layers = {self.layerSizes}\n"
            f"Initial biases = {self.b}\n"
            f"Initial weights = {self.w}\n"
        )

    def inputMinibatch(self):
        """
        Reads csv file with data line by line (each line is a minibatch), converts input "x"s to 1 and "o"s and "b"s to 0, converts the line of data into two tuples of single item strings: the inputs and the theoretical output, and feeds forward each minibatch's inputs into the network.
        """
        with open("ticTacToeData.csv", "r", newline='') as dataFile:
            # non-subscriptable objects aren't containers and don't have indices
            # for minibatch in dataFile:  # each row begins as string
            minibatch = dataFile.readline()
            print(minibatch)  # debugging
            minibatchSplit = minibatch.split(",")  # split string into list
            minibatchInputs = minibatchSplit[0:8]
            for i in range(len(minibatchInputs)):
                if minibatchInputs[i] == "x":
                    minibatchInputs[i] = 1.0
                else:  # if o or b
                    minibatchInputs[i] = 0.0
            minibatchInputs = tuple(minibatchInputs)
            print(minibatchInputs)  # debugging
            theoreticalOutput = tuple(minibatchSplit[9])
            experimentalSigmoidOutputs = []
            # for inputNum in range(len(minibatchInputs)):
            inputNum = 0  # one input
            inputOutput = self.feedforward(
                minibatchInputs[inputNum],                                 inputNum)  # output for one input
            # add to list of outputs for all inputs
            experimentalSigmoidOutputs.append(inputOutput)
            print(experimentalSigmoidOutputs)

    def feedforward(self, neuronInput, inputNum):
        """
        Runs a neuron input through all layers in a network and returns an output for the neuron. Each index in the minibatchInputs list corresponds to a neuron in the input and hidden layers.
        """
        sigmoidOutputList = []  # outputs through all neurons for one input put into network
        # for neuron in range(self.numLayers-1):
        neuron = 0  # one input through one neuron
        print(self.w[neuron])  # debugging
        neuronOutput = self.sigmoid(
            np.dot(self.w[neuron], neuronInput)+self.b[neuron])
        sigmoidOutputList.append(neuronOutput)
        return sigmoidOutputList

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fires or not.
        """
        sigmoidOutput = 1/(1+(math.e**((-1)*dotProdSum)))
        return sigmoidOutput


def main():
    # inputNuerons = int(input("How many inputs do you have? \n"))
    inputNuerons = 9  # debugging
    # outputNuerons = int(input("How many outputs do you want? \n"))
    outputNuerons = 1  # debugging
    neuronsPerLayer = [inputNuerons, inputNuerons, outputNuerons]
    # not sure how to call init() in network class
    network1 = network(neuronsPerLayer)
    network1.inputMinibatch()


if __name__ == main():
    import doctest
    doctest.testmod()
    main()
