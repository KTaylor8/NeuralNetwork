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
        This runs automatically to initialize the attributes for an instance of a class when the instance is created. It takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the NumPy arrays of weights and biases.
        """
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        # lists in which each element is an array for each layer, which each contain the connections/neurons for that layer: weight for each connection (90) and a bias for each hidden and output neuron (10)
        self.w = [np.random.rand(y, x)
                  for x, y in zip(layerSizes[:-1], layerSizes[1:])]
        self.b = [np.random.rand(y, 1) for y in layerSizes[1:]]
        # self.numW = sum(np.prod((self.w[i]).shape) for i in range(len(self.w)))
        # self.numB = sum(np.prod((self.b[i]).shape) for i in range(len(self.b)))
        # print(  # debugging
        #     f"Number of layers = {self.numLayers}\n"
        #     f"Sizes of layers = {self.layerSizes}\n"
        #     f"Initial weights = {self.w}\n"
        #     f"# w = {self.numW}\n"
        #     f"Initial biases = {self.b}\n"
        #     f"# b = {self.numB}\n"
        # )

        # self.w = [
        #     random.uniform(0, 1) for i in range(
        #         layerSizes[0]*layerSizes[1] +  # connections: input + hidden
        #         layerSizes[1]*layerSizes[2])  # connections: hidden + output
        # ]
        # self.b = [
        #     random.uniform(0, 1) for i in range(sum(layerSizes[1:]))
        # ]  # each neuron after the input layer

    def inputMinibatch(self):
        """
        Reads csv file with data line by line (each line is a minibatch), converts input "x"s to 1 and "o"s and "b"s to 0, converts the line of data into two tuples of single item strings: the inputs and the theoretical output, and feeds forward each minibatch's inputs into the network.
        """
        with open("ticTacToeData.csv", "r", newline='') as dataFile:
            # non-subscriptable objects aren't containers and don't have indices
            # for minibatch in dataFile:  # each row begins as string
            minibatch = dataFile.readline()  # need to iterate over all lines
            # print(f'minibatch: {minibatch}')  # debugging
            minibatchSplit = minibatch.split(",")  # split string into list
            minibatchInputs = minibatchSplit[0:8]
            for i in range(len(minibatchInputs)):
                if minibatchInputs[i] == "x":
                    minibatchInputs[i] = 1.0
                else:  # if o or b
                    minibatchInputs[i] = 0.0
            minibatchInputs = tuple(minibatchInputs)
            # print(f'minibatchInputs: {minibatchInputs}')  # debugging
            theoreticalOutput = tuple(minibatchSplit[9])
            # for inputNum in range(len(minibatchInputs)):
            inputNum = 0  # debugging one input
            expOutput = self.feedforward(
                minibatchInputs)
            # print(f'final exp output: {expOutput}')  # debugging

    def feedforward(self, inList):
        """
        Runs all network inputs through all layers in a network and returns an experimental output for the network. Each index in the minibatchInputs list corresponds to a neuron in the input and hidden layers.
        """
        # for layer in range(len(self.layerSizes)-1):  # num layers #debugging
        layer = 0
        layerOutList = []
        # pass inputs into a receiving neuron for as many receiving neurons as there are in the next layer:
        for receivingN in range(self.layerSizes[layer]):
            neuronOutList = []  # outputs from a neuron when fed all inputs
            # for each input that the receiving neuron receives:
            for inNum in range(len(inList)):
                # for neuron in range(self.layerSizes[layerI+1]): don't think needed
                neuronWs = self.w[layer][receivingN]
                neuronBs = self.b[layer][receivingN][0]
                print(
                    f'receiving neuron #: {receivingN}'
                    f'\t1 hidden w for 1 connection: {neuronWs}'
                    f'\t1 hidden b for 1 connection: {hiddenB}')
                neuronOutArr = self.sigmoid(
                    np.dot(hiddenW, inList)+hiddenB
                )
                neuronOutList.append(neuronOutArr[0])
            inList.append(neuronOutList)
            append(neuronInList)
            # print('hiddenOutputList: ', hiddenOutputList)  # debugging

        rawExpOut = inList
        # # for loop not really necessary since we only have 1 output neuron
        # for neuron in range(self.layerSizes[layerI+1]):
        #     print(f"Initial weights = {self.w}\n")
        #     print(f'output layer weights: {self.w[layerI][0]}')
        #     outputW = self.w[layerI][0][neuron]
        #     # print(f"Initial biases = {self.b}\n")
        #     # print(f'output layer bias: {self.b[layerI][0][0]}')
        #     # doesn't change throughout feedforward() b/c there's only 1 output neuron that all connections go through
        #     outputB = self.b[layerI][0][0]
        #     print(
        #         f'neuron #: {neuron}'
        #         f'\t1 hidden w for 1 connection: {hiddenW}'
        #         f'\t1 hidden b for 1 connection: {hiddenB}')
        #     finalNeuronOutputArr = self.sigmoid(
        #         np.dot(outputW, hiddenOutputList)+outputB
        #     )
        #     finalNeuronOutput = finalNeuronOutputArr[0]
        print(f'network raw output: {rawExpOut}')  # debugging
        # expOutput = round(finalNeuronOutput) #commented out for debugging
        # if expOutput == 0:
        #     expOutput == "negative"
        # elif expOutput == 1:
        #     expOutput == "positive"
        # return expOutput #NOT SURE IF I'M SUPPOSED TO HAVE A POSITIVE AND NEGATIVE FOR EVERY NEURON (LIKE IT IS NOW) OR ONLY THE WHOLE NETWORK; IF IT'S THE WHOLE NETWORK, WHAT DO I DO WITH THE OUTPUT FOR EACH INDIVIDUAL INPUT?

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fires or not.
        """
        sigOutput = 1/(1+(math.e**((-1)*dotProdSum)))
        return sigOutput


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
