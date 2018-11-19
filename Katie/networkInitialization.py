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
        self.b = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.w = [np.random.randn(y, x)
                  for x, y in zip(layerSizes[:-1], layerSizes[1:])]  # ?

    def inputMinibatch(self):
        with open("ticTacToeData.csv", newline='') as csvfile:
            readData = csv.reader(csvfile, delimiter=' ')
            for minibatch in readData:  # each row begins as single el list
                minibatchStr = "".join(minibatch)
                minibatchSplit = minibatchStr.split(",")
                theoreticalOutput = tuple(minibatchSplit[9])
                minibatchInputs = tuple(minibatchSplit[0:8])
                for inputNum in minibatchInputs:
                    feedforward(minibatchInputs[inputNum], inputNum)
                    # not sure how to assign number values to the inputs to make them calculatable in the sigmoid function; and not sure how to call the feedforward method

    def feedforward(self, neuronInput, inputNum):
        """Runs a neuron input through all layers in a network and returns an output for the neuron"""
        # I'M CONFUSED ABOUT HOW TO DO THE LOOPING SO THAT IT GOES THROUGH ALL NEURONS AND ALL LAYERS
        # each input corresponds to a neuron in the hidden layer
        for neuron in zip(self.b, self.w):
            sigmoidOutput = sigmoid(np.dot(self.w[neuron],
                                           neuronInput)+self.b[neuron])

        # compares sigmoidOutput to threshold to see if it fires or not:
        # I'm not sure what the firing and not firing looks like in code
        if sigmoidOutput > 0.5:
            # fires and passes something on to outputs
        elif sigmoidOutput <= 0.5:
            # doesn't fire
        return neuronOutput  # currently undefined

    """
    The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fiers or not.
    """

    def sigmoid(self, x):
        sigmoidOutput = 1/(1+(math.e**((-1)*x)))
        return sigmoidOutput


if __name__ == main():  # not sure how class relates to main()
    import doctest
    doctest.testmod()
    main()
