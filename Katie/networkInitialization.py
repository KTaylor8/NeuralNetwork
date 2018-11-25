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

    def init(self, layerSizes):
        """
        Takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the weights and biases. The input layer is the first one and the weights and biases aren't applied to it.
        """
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        # self.b = [random.uniform(0, 1) for y in layerSizes[1:]]
        # self.w = [random.uniform(0, 1) for y in layerSizes[1:]]
        self.b = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.w = [np.random.randn(y, x)
                  for x, y in zip(layerSizes[:-1], layerSizes[1:])]  # ?

    def inputMinibatch(self):
        """
        Reads csv file with data line by line (each line is a minibatch), converts input "x"s to 1 and "o"s and "b"s to 0, converts the line of data into two tuples of single item strings: the inputs and the theoretical output, and feeds forward each minibatch's inputs into the network.
        """
        with open("ticTacToeData.csv", newline='') as csvfile:
            readData = csv.reader(csvfile, delimiter=' ')
            for minibatch in readData:  # each row begins as single el list
                minibatchStr = "".join(minibatch)
                minibatchSplit = minibatchStr.split(",")
                minibatchInputs = minibatchSplit[0:8]
                for i in minibatchInputs:
                    if minibatchInputs[i] == "x":
                        minibatchInputs[i] = 1
                    else:  # if o or b
                        minibatchInputs[i] = 0
                minibatchInputs = tuple(minibatchInputs)
                theoreticalOutput = tuple(minibatchSplit[9])
                for inputNum in minibatchInputs:
                    feedforward(minibatchInputs[inputNum], inputNum)
                    # not sure how to call the feedforward method

    def feedforward(self, neuronInput, inputNum):
        """
        Runs a neuron input through all layers in a network and returns an output for the neuron. Each index in the minibatchInputs list corresponds to a neuron in the input and hidden layers.
        """
        sigmoidOutputList = []
        for neuron in zip(self.b, self.w):
            sigmoidOutputList.append(sigmoid(np.dot(self.w[neuron],
                                                    neuronInput)+self.b[neuron]))

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fires or not.
        """
        sigmoidOutput = 1/(1+(math.e**((-1)*dotProdSum)))
        return sigmoidOutput


if __name__ == main():  # not sure how class relates to main()
    import doctest
    doctest.testmod()
    main()
