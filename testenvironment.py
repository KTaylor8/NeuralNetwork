import math
import os
import numpy as np
import matplotlib

class network():



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
        print(nablaW)
        time.sleep(5)
        nablaB = [nablaB + deltaNablaB for nablaB,
                  deltaNablaB in zip(nablaB, deltaNablaB)]
        print(nablaB)
        time.sleep(5)
        self.w = [w - (self.learningRate/(self.layerSizes[0]+1)) *
                  nablaW for w, nablaW in zip(self.w, nablaW)]
        #print(self.w)
        self.b = [b - (self.learningRate/(self.layerSizes[0]+1)) *
                  nablaB for b, nablaB in zip(self.b, nablaB)]
        #print(self.b)

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

        # error and output change calculations
        error = self.costDerivative(
                activations[-1], expOut) * self.sigmoidPrime(weightedSumList[-1])
        nablaB[-1] = error
        nablaW[-1] = np.dot(error, activations[-2].transpose())

        # backpropagate error using output error
        # find change in weights and biases for entire network
        for L in range(2, len(self.layerSizes)):
            weightedSum = weightedSumList[-L]
            sp = self.sigmoidPrime(weightedSum)
            error = np.dot(self.w[-L+1].transpose(), error) * sp
            # print(nablaB[-L])
            nablaB[-L] = error
            # print(f"nablaB array for layer {-L}: {nablaB[-L]}")
            nablaW[-L] = np.dot(error, activations[-L-1].transpose())
            # print(f"nablaW array for layer {-L}: {nablaW[-L]}")
        
        # print(f"nablaW: {nablaW}")
        # print(f"nablaB: {nablaB}")

        return nablaB, nablaW

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
