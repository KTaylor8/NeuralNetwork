import csv
import os
import random
import numpy as np
import math

# asks user to set number of layers and neurons and outputs (9,9,3)
# randomizes weights and biases
# setting minibatch (row) that runs through each time and changes weights and biases based on accuracy


class network():

    def init(self, layerSizes):
        """Takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the weights and biases. The input layer is the first one and the weights and biases aren't applied to it."""
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        # self.b = [random.uniform(0, 1) for y in layerSizes[1:]]
        # self.w = [random.uniform(0, 1) for y in layerSizes[1:]]
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w = [np.random.randn(y, x)
                  for x, y in zip(layerSizes[:-1], sizes[1:])]  # ?

    def inputMinibatch(self):
        with open("ticTacToeData.csv", newline='') as csvfile:
            readData = csv.reader(csvfile, delimiter=' ')
            for minibatch in readData:  # each row begins as single el list
                minibatchStr = "".join(minibatch)
                minibatchSplit = minibatchStr.split(",")
                # print(minibatchSplit)
                theoreticalOutput = tuple(minibatchSplit[9])
                minibatchInputs = tuple(minibatchSplit[0:8])
                for i in minibatchInputs:
                    # not sure how to assign number values to the inputs to make them calculatable in the sigmoid function
                    feedforward(minibatchInputs[i])
                # break

    def feedforward(self, neuronInput):
        """Runs a neuron input through all layers in a network and returns an output for it"""
        for b, w in self.b, self.w:
            neuronOutput = sigmoid(np.dot(w, neuronInput)+b)

        dotProdSum = sumDotProd(f, w, b)
        sigmoidOutput = sigmoid(dotProdSum)

        # compares sigmoidOutput to threshold to see if it fires or not:
        if sigmoidOutput > 0.5:
            print('1')
        if sigmoidOutput <= 0.5:
            print('0')
        return neuronOutput

    """
    The sigmoid activation function will map the raw output of inputs, weights, and biases onto a function, which can then be used to determine if a neuron fires or not.
    """

    def sigmoid(x):
        neuronOutput = 1/(1+(math.e**((-1)*x)))
        return neuronOutput

    """
    The next step is to define the raw output function. This will take the sum of the dot product of the weights and inputs, then adds the bias.

    A dot product is the product of two vectors (weights and inputs)
    """

    def sumDotProd(inputMinibatch, w, b):
        dotProd = np.multiply(inputMinibatch, w)
        dotProdSum = np.sum(dotProd, axis=None) + b
        return dotProdSum


if __name__ == main():
    import doctest
    doctest.testmod()
    main()
