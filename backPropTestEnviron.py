import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import time


class network():
    def __init__(self, layerSizes, learningRate):
        self.layerSizes = layerSizes
        self.learningRate = learningRate
        self.output = [0]
        self.inputs = [1, 2, 3, 4, 3, 0.5, 2, 3, 2]
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
        # print(self.b)

    def runbackprop(self):
        expOut = 0.5
        inputs = self.inputs
        self.updateWB(inputs, expOut)

    def updateWB(self, inputs, expOut):
        """
        Updates the weights and biases of the network based on the partial derivatives of the cost function. Variables are self (class specific variable), the list miniBatch, and the learning rate
        """

        nablaW = [np.zeros(layer.shape) for layer in self.w]
        # print(nablaW)
        nablaB = [np.zeros(layer.shape) for layer in self.b]
        # print(nablaB)

        deltaNablaB, deltaNablaW = self.backprop(
            inputs, expOut)

        nablaW = [nablaW + deltaNablaW for nablaW,
                  deltaNablaW in zip(nablaW, deltaNablaW)]
        # print(nablaW)
        # time.sleep(1)

        nablaB = [nablaB + deltaNablaB for nablaB,
                  deltaNablaB in zip(nablaB, deltaNablaB)]
        # print(nablaB)
        time.sleep(1)

        self.w = [w - (self.learningRate/(self.layerSizes[0]+1)) *
                  nablaW for w, nablaW in zip(self.w, nablaW)]
        # print(self.w)
        self.b = [b - (self.learningRate/(self.layerSizes[0]+1)) *
                  nablaB for b, nablaB in zip(self.b, nablaB)]
        # print(self.b)

    def backprop(self, inputs, expOut):
        """
        Uses feedforward of network to calculate error for output layer, uses that to backpropagate error to other layers, and finally find the change in weights and biases based on the errors
        """
        nablaW = [np.zeros(layer.shape) for layer in self.w]
        nablaB = [np.zeros(layer.shape) for layer in self.b]
        activation = inputs
        activations = [inputs]
        weightedSumList = []
        # feedforward
        for bArray, wArray in zip(self.b, self.w):  # layers/arrays = 2
            weightedSum = np.dot(wArray, inputs)+bArray
            weightedSumList.append(weightedSum)
            # print(weightedSumList)
            activation = self.sigmoid(weightedSum)
            activations.append(activation)
            # print(activations)
        print(f"weightedSumList: {weightedSumList}")

        # error and output change calculations
        error = self.costDerivative(
            activations[-1], expOut) * self.sigmoidPrime(weightedSumList[-1])
        nablaB[-1] = error
        nablaW[-1] = np.dot(error, activations[-1].transpose())
        # print(f"last layer error: {error}")

        # backpropagate error using output error
        # find change in weights and biases for entire network
        for L in range(2, len(self.layerSizes)):
            weightedSum = weightedSumList[-L]
            print(f"weighted sum: {weightedSum}")
            sp = self.sigmoidPrime(weightedSum)
            print(f"sigmoid prime: {sp}")
            error = np.dot(self.w[-L+1].transpose(), error) * sp
            # print(f"error for layer {-L}: {error}")

            # switched order in dot prod arguments for debugging
            nablaB[-L] = error
            print(f"nablaB array for layer {-L}: {nablaB[-L]}")
            nablaW[-L] = np.dot(error, activations[-L].transpose())
            # print(f"nablaW array for layer {-L}: {nablaW[-L]}")

        # print(f"nablaW: {nablaW}")
        # print(f"nablaB: {nablaB}")

        return nablaB, nablaW

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine the output array of the layer.
        """
        activation = 1/(1+np.exp(-dotProdSum))
        return activation

    def sigmoidPrime(self, s):
        """
        Function for the derivative of the activation function. Used to find the error of each neuron
        """
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def costDerivative(self, expOut, tOut):
        """
        Function for the derivative of the cost function. Used to find the error of each neuron
        """
        networkOut = np.array(expOut, dtype='float64')
        y = np.array(tOut, dtype='float64')
        costPrime = np.subtract(networkOut, y)
        return costPrime


def main():
    """
    This function sets the sizes of the layers and the learning rate of the network.
    """
    inputNuerons = 9  # debugging
    outputNuerons = 1  # debugging
    neuronsPerLayer = [inputNuerons, inputNuerons, outputNuerons]
    learningRate = 1  # debugging
    network1 = network(neuronsPerLayer, learningRate)
    return network1.runbackprop()


if __name__ == "__main__":
    main()
