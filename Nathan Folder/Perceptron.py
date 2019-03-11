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

        for num in range(60):
            self.w.append(random.uniform(-1, 1))

        self.b.append(random.uniform(-1, 1))
        # not generating weights and biases
        # print(self.w)
        # print(self.b)

    def readCSV(self, datafile):

        minibatches = []
        for minibatch in datafile:
            minibatchsplit = minibatch.strip().split(",")
            for i in range(len(minibatchsplit)-1):
                minibatchsplit[i] = float(minibatchsplit[i])
                if minibatchsplit[-1] == 'M':  # theoretical output
                    minibatchsplit[-1] = 1.0
                elif minibatchsplit[-1] == 'R':
                    minibatchsplit[-1] = 0.0
            minibatches.append(minibatchsplit)
        return minibatches

    def runPerceptron(self):
        with open(
            #r"sonar.all-data.csv"
             r"C:\Users\s-2508690\Desktop\NeuralNetwork\Nathan Folder\sonar.all-data.csv"
        ) as datafile:
            miniBatchNum = 1
            accuracyRates = []
            numCorrect = 0
            minibatches = self.readCSV(datafile)

            for epoch in range(self.epochs):

                for minibatch in minibatches:
                    inputs = minibatch[0:self.layerSizes[0]]
                    self.reqOutput = minibatch[-1]
                    preOutput = self.feedforward(inputs)
                    self.updateWB(inputs, preOutput)

                    if preOutput == self.reqOutput:
                        numCorrect = numCorrect + 1

                    groupsOf = 50
                    if miniBatchNum % groupsOf == 0:
                        percentsCorrect = float(
                            round((numCorrect/groupsOf)*100)
                        )
                        accuracyRates.append(percentsCorrect)
                        numCorrect = 0

                    miniBatchNum = miniBatchNum + 1

            print(f"Accuracy Rates: {accuracyRates}")
            return accuracyRates

    def updateWB(self, inputs, preOutput):

        nablaW = []
        nablaB = []

        deltaNablaW, deltaNablaB = self.backprop(inputs, preOutput)
        nablaB = deltaNablaB
        nablaW = deltaNablaW
        # trying to concentate generate with list
        self.w = np.add(self.w, nablaW)
        self.b = np.add(self.b, nablaB)
        # print("The new weights are", self.w)
        # print("The new biases are", self.b)
        # time.sleep(1)

    def backprop(self, inputs, reqOutput):

        # feedforward
        activations = []
        dotProduct = np.dot(inputs, self.w) + self.b
        activation = self.step(dotProduct)
        activations.append(activation)

        # stochastic gradient descent
        error = self.costderivative(activations[-1])
        nablaW = list((error * self.learningRate * i) for i in inputs)
        nablaB = (error * self.learningRate)
        # print(f"the changes in biases are {nablaB}")
        # print(f"the changes in weights are {nablaW}")
        return nablaW, nablaB

    def feedforward(self, inputs):

        dotProduct = np.dot(inputs, self.w) + self.b
        preOutput = self.step(dotProduct)
        return preOutput

    def step(self, dotProduct):
        preOutput = 0.0
        if dotProduct >= 0.0:
            preOutput = 1.0
        return preOutput

    # something with sigmoid function reduces the scale of changes
    def sigmoid(self, dotProduct):
        preOutput = 1/(1+math.exp(-dotProduct))
        return preOutput

    def costderivative(self, preOutput):
        return (self.reqOutput - preOutput)


def main():
    inputneurons = 60
    hiddenneurons = 1
    outputneurons = 1
    layerSizes = [inputneurons, hiddenneurons, outputneurons]
    learningRate = 0.14
    epochs = 10
    runnetwork = perceptron(layerSizes, learningRate, epochs)
    return runnetwork.runPerceptron()


def graphUpdate(frame):
    xdata.append(frame)
    try:
        ydata.append(percentagesCorrect[int(frame-1)])
    except IndexError:
        pass
    # Set the x and y data; ACCEPTS: 2D array (rows are x, y) or two 1D arrays
    # print(ydata)
    graphLine.set_data(xdata, ydata)
    return graphLine,


if __name__ == "__main__":

    percentagesCorrect = main()
    numIterationsLimit = len(percentagesCorrect)+1

    plt.switch_backend('TkAgg')

    # initialize graph:
    fig, ax = plt.subplots()
    graphLine, = plt.plot([], [], 'r-', animated=True)
    redPatch = mpatches.Patch(color='red', label='Network')
    plt.legend(handles=[redPatch], loc="lower right")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage Correct")
    plt.title("Percentage Correct Over Time")
    # ([x start, x end, y start, y end])
    plt.axis([1, numIterationsLimit, 0, 101])

    ticksList = []
    for i in range(1, numIterationsLimit):
        ticksList.append(i)
    ax.set_xticks(ticksList)

    xdata, ydata = [], []
    ani = animation.FuncAnimation(fig,
                                  graphUpdate,
                                  frames=np.linspace(
                                      1,
                                      numIterationsLimit,
                                      num=numIterationsLimit-1
                                  ),
                                  blit=True
                                  )
    plt.show()
