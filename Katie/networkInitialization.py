# enter into terminal:
# import network
# net = network.Network([784, 30, 10])

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
        self.b = random.uniform(0, 1)
        self.w = random.uniform(0, 1)
        #self.b = [np.random.randn(y, 1) for y in layerSizes[1:]]
        # self.w = [np.random.randn(y, x)
        #           for x, y in zip(layerSizes[:-1], layerSizes[1:])]

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
                    feedforward(minibatchInputs[i])
                # break

    def feedforward(self, neuronInput):
        """Runs a neuron input through all layers in a network and returns an output for it"""
        for b, w in zip(self.b, self.w):
            neuronOutput = sigmoid(np.dot(w, neuronInput)+b)
        return neuronOutput

    """
    The sigmoid activation function will map the raw output of inputs, weights, and biases onto a function, which can then be used to determine if a neuron fires or not.
    """

    def sigmoid(x):
        neuronOutput = 1/(1+(math.e**((-1)*x)))
        return neuronOutput

    # """
    # The next step is to define the raw output function. This will take the sum of the dot product of the weights and inputs, then adds the bias.

    # A dot product is the product of two vectors (weights and inputs)
    # """

    # def __sumofdot(x, w, b):
    #     f1 = np.multiply(x, w)
    #     f2 = np.sum(f1, axis=None) + b
    #     return f2

    def main():
        """ Creates a list of inputs from the user, and defines the length of the list of inputs
        """

        f = np.array([float(x) for x in input("enter list: ").split()])
        l = f.size

        """
        Creates a list of weights for every input, and creates a set bias
        """

        w = np.random.randint(0, high=3, size=(1, l))
        print("The weights are", w)
        b = -5

        """calls the sum of the dot product and the activation function, which uses the list of inputs, weights, and bias to find the raw output
        """

        f2 = __sumofdot(f, w, b)
        print("The sum of the dot product is", f2)

        f3 = sigmoid(f2)
        print("The raw output is", f3)

        """Sets a threshold for the activation function to determine if it fires or not (This may or may not be taken out, depending on how the neural network end up)
        """

        if f3 > 0.5:
            print('1')
        if f3 <= 0.5:
            print('0')


def main():
    # not sure how to call the class's methods at the right time yet
    numInputs = input("How many input neurons will this network have?\n")
    numOutputs = input("How many output neurons will this network have?\n")
    layerSizes = [numInputs, numInputs, numOutputs]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
