import os
import random
import numpy as np
import math


class network():

    def __init__(self, layerSizes, learningRate):
        """
        This runs automatically to initialize the attributes for an instance of a class when the instance is created. It takes in list layerSizes that has the number of neurons per layer and uses it to determine the number of layers and randomize the NumPy arrays of weights and biases.
        """
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.learningRate = learningRate
        # lists in which each element is an array for each layer, which each contain the connections/neurons for that layer: weight for each connection (90) and a bias for each hidden and output neuron (10)

        allWList = []
        allBList = []

        for layer in range(len(layerSizes)-1):
            wInLayerList = []

            for receivingN in range(layerSizes[layer+1]):
                wForNeuronList = []

                for givingN in range(layerSizes[layer]):
                    wForNeuronList.append(random.uniform(-1, 1))

                wInLayerList.append(wForNeuronList)

            wInLayerArray = np.reshape(
                (np.asarray(wInLayerList)),
                (self.layerSizes[layer+1], self.layerSizes[layer])
            )
            allWList.append(wInLayerArray)

        self.w = allWList

        for layer in range(len(layerSizes)-1):
            bInLayerList = []

            for neuron in range(layerSizes[layer+1]):
                bInLayerList.append(random.uniform(-1, 1))

            bInLayerArray = np.reshape(
                (np.asarray(bInLayerList)),
                (self.layerSizes[layer+1], 1)
            )
            allBList.append(bInLayerArray)

        self.b = allBList

        # alternate generation code limited to range [0, 1):
        # self.b = [np.random.rand(y, 1) for y in layerSizes[1:]]
        # self.w = [np.random.rand(y, x)
        #       for x, y in zip(layerSizes[:-1], layerSizes[1:])])

    def runNetwork(self, learningRate, testData=None):
        """This part of the program will 
        create the miniBatch from the epoch
        run the gradient descent on that         
        repeat for remaining epoch
        switch to next epoch
        End program when test data runs out"""

        with open(
                "ticTacToeData.csv", "r", newline=''
        ) as dataFile:
            # non-subscriptable objects aren't containers and don't have indices
            minibatches, inputs = self.makeMinibatchesList(dataFile)
            for minibatch in range(len(minibatches)):
                expOutput, zList, activations = self.feedforward(inputs)
                # print(expOutput)
                tOutput = tuple(minibatchSplit[9])
                self.updateWB(minibatch, learningRate,
                              activations, tOutput, zList)

            """Still working on how to end an epoch/exit out of program"""

    def makeMinibatchesList(self, dataFile):
        minibatches = []
        for minibatch in dataFile:  # each row begins as string
            minibatchSplit = minibatch.strip().split(",")
            minibatchInputs = minibatchSplit[0:9]  # end is exclusive
            for i in range(len(minibatchInputs)):
                if minibatchInputs[i] == "x":
                    minibatchInputs[i] = 1.0
                else:  # if o or b
                    minibatchInputs[i] = 0.0
            inputs = np.reshape(
                (np.asarray(minibatchInputs)),
                (self.layerSizes[0], 1)
            )  # (rows, columns)
            minibatches.append(minibatch)
        return minibatches, minibatchInputs

    def feedforward(self, inputs):
        """Return the output of the network if the inList of inputs is received."""
        neuronOuts = []
        zList = []
        for bArray, wArray in zip(self.b, self.w):  # layers/arrays = 2
            z = (np.dot(wArray, inputs)+bArray)
            zList.append(z)
            activation = self.sigmoid(z)
            for output in range(len(activation)):
                neuronOuts.append(activation[output])
            inputs = activation
            # 1st iteration returns an array of 9 single element lists
            # break

        # expOut = np.sign(rawOut[0][0]) #threshold based on sign, but always +
        # if expOut == 1.0 or expOut == 0.0:  # NOT SURE WHAT TO DO IF == 0
        #     expOut = "positive"
        # elif expOut == -1.0:
        #     expOut = "negative"

        expOut = round(activation[0][0])  # threshold based on rounding
        if expOut == 1.0:
            expOut = "positive"
        elif expOut == 0.0:
            expOut = "negative"
        #print(f'rawOut: {rawOut[0][0]}\tsign: {expOut}')
        return expOut, zList, neuronOuts

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fires or not.
        """
        activation = 1/(1+(math.e**((-1)*dotProdSum)))
        return activation

    def updateWB(self, inputs, learningRate, activations, tOutput, zList):
        """Updates the weights and biases of the network based on the partial derivatives of the cost function. Variables are self (class specific variable), the list miniBatch, and the learning rate"""

        nablaW = (np.zeros(self.w[0].shape) for w in self.w)
        print(nablaW)
        nablaB = (np.zeros(self.b[0].shape) for b in self.b)
        print(nablaB)

        for [inputs, tOutput] in minibatch:
            deltaNablaB, deltaNablaW = self.backprop(
                inputs, tOutput, activations, zList, self.layerSizes)

            nablaW = (nablaW + deltaNablaW for nablaW,
                      deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB,
                      deltaNablaB in zip(nablaB, deltaNablaB))

            self.w = (w - (learningRate/len(minibatch)) *
                      nablaW for w, nablaW in zip(self.w, nablaW))

            self.b = (b - (learningRate/len(minibatch)) *
                      nablaB for b, nablaB in zip(self.b, nablaB))

    def backprop(self, inputs, tOutput, activations, zList, layerSizes):
        """
        Feedforward section of the network. Calculates the activation for each neuron of the network and 
        """
        deltaNablaW = np.zeros(self.w[0].shape)
        print(deltaNablaW)
        deltaNablaB = np.zeros(self.b[0].shape)
        print(deltaNablaB)

        zList = []

        # error and output change calculations
        numLayers = self.layerSizes
        error = self.costDerivative(
            activations[-1], expOutput) * self.sigmoidPrime(zList[-1])
        deltaNablaB[-1] = error
        deltaNablaW[-1] = np.dot(error, activations[-2].transpose)

        # other error calculations
        deltaNablaB = error
        deltaNablaW = np.dot(error, activations[numLayers - 1].transpose())

        for l in range(2, self.layerSizes):
            z = zList[-1]
            sp = self.sigmoidPrime(z)
            error = np.dot(self.w[-l+1].transpose(), error) * sp
            deltaNablaB[-1] = error
            deltaNablaW[-1] = np.dot(error, activations[-l-1].transpose())

        return deltaNablaB, deltaNablaW

    def sigmoidPrime(self, s):
        """
        Function for the derivative of the activation function. Used to find the error of each neuron
        """
        sp = (self.sigmoid(s) - self.sigmoid(s)**2)
        return sp

    def costDerivative(self, expOutput, tOutput):
        """
        Function for the derivative of the cost function. Used to find the error of each neuron
        """
        return (expOutput - tOutput)


def main():
    # inputNuerons = int(input("How many inputs do you have? \n"))
    inputNuerons = 9  # debugging
    # outputNuerons = int(input("How many outputs do you want? \n"))
    outputNuerons = 1  # debugging
    neuronsPerLayer = [inputNuerons, inputNuerons, outputNuerons]
    #learningRate = float(input("What's the learning rate \n"))
    learningRate = 0.01  # debugging
    # not sure how to call init() in network class
    network1 = network(neuronsPerLayer, learningRate)
    network1.runNetwork(learningRate)


if __name__ == main():
    import doctest
    doctest.testmod()
    main()
