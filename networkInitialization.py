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

    def inputMinibatch(self):
        """
        Reads csv file with data line by line (each line is a minibatch), converts input "x"s to 1 and "o"s and "b"s to 0, converts the line of data into two tuples of single item strings: the inputs and the theoretical output, and feeds forward each minibatch's inputs into the network.
        """
        with open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\Nathan Folder\ticTacToeData.csv", "r", newline='') as dataFile:
            # non-subscriptable objects aren't containers and don't have indices
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
                expOutput = self.feedforward(inputs)
                # print(expOutput)
                tOutput = tuple(minibatchSplit[9])

    def feedforward(self, inputs):
        """Return the output of the network if the inList of inputs is received."""
        for bArray, wArray in zip(self.b, self.w):  # layers/arrays = 2
            rawOut = self.sigmoid(np.dot(wArray, inputs)+bArray)
            inputs = rawOut  # my addition to account for the 2 layer spaces
            # 1st iteration returns an array of 9 single element lists
            # break

        # expOut = np.sign(rawOut[0][0]) #threshold based on sign, but always +
        # if expOut == 1.0 or expOut == 0.0:  # NOT SURE WHAT TO DO IF == 0
        #     expOut = "positive"
        # elif expOut == -1.0:
        #     expOut = "negative"

        expOut = round(rawOut[0][0])  # threshold based on rounding
        if expOut == 1.0:
            expOut = "positive"
        elif expOut == 0.0:
            expOut = "negative"
        #print(f'rawOut: {rawOut[0][0]}\tsign: {expOut}')
        return expOut

    def sigmoid(self, dotProdSum):
        """
        The sigmoid activation function put the inputs, weights, and biases into a function that helps us determine if the neuron fires or not.
        """
        sigOutput = 1/(1+(math.e**((-1)*dotProdSum)))
        return sigOutput

    def sigmoidprime(self, s):
        """Function for the derivative of the activation function. Used to find the error of each neuron"""
        sp = (self.sigmoid(s) - self.sigmoid(s)**2)
        return sp

    def costderivative(self, expOutput, tOutput):
        """Function for the derivative of the cost function. Used to find the error of each neuron"""
        return (expOutput - tOutput)

    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData
    = None):
        """This part of the program will 
        create the miniBatch from the epoch
        run the gradient descent on that miniBatch
        repeat for remaining epoch
        switch to next epoch
        End program when test data runs out"""
        n = len(trainingData)
        for k in range(epochs):
            miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
        for miniBatch in miniBatches:
            self.updateWB(miniBatch, learningRate, self.w, self.b)

            """Still working on how to end an epoch/exit out of program"""

        if testData:
            print ("Epoch Over")

    def backprop(self, inputs, tOutput, numLayers):
            """This function calculates the rate of change of the cost function and the biases/weights, uses that to find the error of each neuron, and uses the error to calculate the change in weights and biases. Variables are self, the change in b and a, and the number of layers in a minibatch"""
            #define activation, weighted inputs, and set up all lists/tuples
            
            """Feedforward section of the network. Calculates the activation for each neuron of the network and """
            for bArray, wArray in zip(self.b, self.w):  # layers/arrays = 2
                rawOut = self.sigmoid(np.dot(wArray, inputs)+bArray)
                inputs = rawOut

            errorL = self.costderivative((activations[numLayers], tOutput)) * self.sigmoidprime(z[numLayers])

            deltaNablaB = errorL
            deltaNablaW = np.dot(errorL, activations[numLayers - 1].transpose())

            for l in range(2, numLayers):
                z = zList[numLayers]
                sp = self.sigmoidprime(z)
                errorL = np.dot(self.w[-l+1].transpose(), errorL) * sp
                nablaB = errorL
                nablaW = np.dot(errorL, activations[-l-1].transpose())

            return deltaNablaB, deltaNablaW

    def updateWB(self, inputs, learningRate, w, b):
        """Updates the weights and biases of the network based on the partial derivatives of the cost function. Variables are self (class specific variable), the list miniBatch, and the learning rate"""
    
        nablaW = np.zeros(w[0].shape)
        print(nablaW)
        nablaB = np.zeros(b[0].shape)
        print(nablaB)

        for [inputs, tOutput] in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(inputs, tOutputs, numLayers)

            nablaW = (nablaW + deltaNablaW for nablaW, deltaNablaW in zip(nablaW, deltaNablaW))

            nablaB = (nablaB + deltaNablaB for nablaB, deltaNablaB in zip(nablaB, deltaNablaB))

            self.w = (w - (learningRate/len(miniBatch))*nablaW for w, nablaW in zip(w, nablaW))

            self.b = (b - (learningRate/len(miniBatch))*nablaB for b, nablaB in zip(b, nablaB))



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
