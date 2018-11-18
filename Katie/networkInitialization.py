# enter into terminal:
# import network
# net = network.Network([784, 30, 10])

import csv
import os
import random
import numpy as np

# asks user to set number of layers and neurons and outputs (9,9,3)
# randomizes weights and biases
# setting minibatch (row) that runs through each time and changes weights and biases based on accuracy


class network():

    def init(self, layerSizes):
        """Takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the weights and biases."""
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.biases = [np.random.randn(y, 1) for y in layerSizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layerSizes[:-1], layerSizes[1:])]

    def inputData(self):
        with open("ticTacToeData.csv", newline='') as csvfile:
            readData = csv.reader(csvfile, delimiter=' ')
            for minibatch in readData:  # each row is a single element list
                minibatchStr = "".join(minibatch)
                minibatchSplit = minibatchStr.split(",")
                print(minibatchSplit)
                theoreticalOutput = tuple(minibatchSplit[9])
                minibatchInputs = tuple(minibatchSplit[0:8])
                print(minibatchInputs)
                break

    def feedforward(self, neuronInput):
        """Return the output of the network for an input 'neuronInput' """
        for b, w in zip(self.biases, self.weights):
            neuronOutput = sigmoid(np.dot(w, neuronInput)+b)
        return neuronOutput


def main():
    # not sure how to call the class's methods at the right time yet
    numInputs = input("How many input neurons will this network have?\n")
    numOutputs = input("How many output neurons will this network have?\n")
    layerSizes = [numInputs, numInputs, numOutputs]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
