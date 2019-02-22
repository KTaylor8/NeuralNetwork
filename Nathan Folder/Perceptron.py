import math
import time
import os
import random
import numpy as np
import matplotlib

class perceptron:
    def __init__(self):
        self.inputs = [0, 1]
        self.outputs = [-1, 1, 1]
        self.learningRate = 0.01
        self.layerSizes = [2, 1, 1]

        self.w = [0.5, 0.5]
        self.b = [1]
        
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
        print(f"Initial Weights: {self.w}")

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
        
        print(f"Network Biases: {self.b}")
    def runNetwork(self, inputs):
        output = self.feedforward(inputs)
        self.updateWB(inputs, output)

    def feedforward(self, inputs):
        dotProduct = (self.w*inputs)+self.b
        output = sigmoid(dotProduct)

    def sigmoid(self, dotProduct):
        activation = 1/(1+np.exp((-1)*dotProduct))
        return activation