import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

class perceptron():

    def __init__(self, layerSizes, learningRate, epochs):

        self.w = []
        self.b = []
        self.layerSizes = layerSizes
        self.learningRate = learningRate
        self.epochs = epochs
        inputN = layerSizes[1]

        for n in range((inputN - 1)):
            self.w.append(random.uniform(-1, 1))

        self.b.append(random.uniform(-1, 1))
            #print(self.w)
            #print(self.b)

    def readCSV(self, datafile):

        minibatches = []
        for minibatch in datafile:
            minibatchsplit = minibatch.strip().split(",")
            for i in range(len(minibatchsplit)-1):
                if minibatchsplit[9] == 'M':  # theoretical output
                    minibatchsplit[9] = 1.0
                elif minibatchsplit[9] == 'R':
                    minibatchsplit[9] = 0.0
            minibatches.append(minibatchsplit)
        return minibatches

    def runPerceptron(self):

        with open (r"C:\Users\s-2508690\Desktop\NeuralNetwork\sonar.all-data.csv") as datafile:
            miniBatchNum = 1
            accuracyRates = []
            for i in range(self.epochs):
                minibatches = self.readCSV(datafile)

                for minibatch in minibatches:
                    inputs = minibatch[0:3]
                    self.reqOutput = minibatch[4]
                    preOutput = self.feedforward(inputs)
                    self.updateWB(inputs, preOutput)
                    miniBatchNum = miniBatchNum + 1

                if preOutput == self.reqOutput:
                    numCorrect = numCorrect + 1

                groupsOf = 50
                if miniBatchNum % groupsOf == 0:
                    percentsCorrect = float(
                        round((numCorrect/groupsOf)*100)
                    )
                    accuracyRates.append(percentsCorrect)
                    numCorrect = 0
                minibatchNum = minibatchNum + 1
            print(f"Accuracy Rates: {accuracyRates}") 
            return accuracyRates

    def updateWB(self, inputs, preOutput):

        nablaW = []
        nablaB = []

        deltaNablaW, deltaNablaB = self.backprop(inputs, preOutput)
        self.w = self.w + nablaW
        self.b = self.b + nablaB
        print("The new weights are", self.w)
        print("The new biases are", self.b)


    def backprop(self, inputs, reqOutput):

        #feedforward
        activations = []
        dotProduct = (inputs * self.w) + self.b
        activation = self.step(dotProduct)
        activations.append(activation)

        #stochastic gradient descent
        error = self.costderivative(activations[-1])
        nablaB = error * self.learningRate * inputs
        nablaW = error * self.learningRate * inputs
        return nablaW, nablaB

    def feedforward(self, inputs):

        dotProduct = np.dot(inputs, self.w) + self.b
        preOutput = self.step(dotProduct)
        return preOutput

    def step(self, dotProduct):

        if dotProduct >= 0.0:
            activation = 1.0
        elif dotProduct > 0.0:
            activation = 0.0
        return activation

    def costderivative(self, preOutput):
        return (self.reqOutput - preOutput)

def main():
    inputneurons = 60
    hiddenneurons = 1
    outputneurons = 1
    layerSizes = [inputneurons, hiddenneurons, outputneurons]
    learningRate = 0.1
    epochs = 2
    runnetwork = perceptron(layerSizes, learningRate, epochs)
    return runnetwork.runPerceptron()

def graphUpdate(frame):
    xdata.append(frame)
    try: 
        ydata.append(percentagesCorrect[int(frame)])
    except IndexError:
        pass
    # Set the x and y data; ACCEPTS: 2D array (rows are x, y) or two 1D arrays
    graphLine.set_data(xdata, ydata)
    return graphLine,

if __name__ == "__main__":

    percentagesCorrect = main()
    numIterations = len(percentagesCorrect)

    plt.switch_backend('TkAgg')

    #initialize graph:
    fig, ax = plt.subplots()
    graphLine, = plt.plot([], [], 'r-', animated=True)
    redPatch = mpatches.Patch(color='red',label='Network')
    plt.legend(handles=[redPatch], loc="upper right")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage Correct")
    plt.title("Percentage Correct Over Time")
    plt.axis([1, numIterations, 0, 100])  # ([x start, x end, y start, y end])

    ticksList = []
    for i in range(1, numIterations):
        ticksList.append(i)
    ax.set_xticks(ticksList)

    xdata, ydata = [], []
    ani = animation.FuncAnimation(fig,
                                  graphUpdate,
                                  frames=np.linspace(
                                    1,
                                    numIterations,
                                    num=numIterations
                                    ),
                                    blit=True
    )
    plt.show()

    
    