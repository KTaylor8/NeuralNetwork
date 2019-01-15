def runNetwork(self, learningRate, testData=None):
    """This part of the program will
    create the miniBatch from the epoch
    run the gradient descent on that
    repeat for remaining epoch
    switch to next epoch
    End program when test data runs out"""

    with open(
            r"C:\Users\s-2508690\Desktop\NeuralNetwork\ticTacToeData.csv", "r", newline=''
    ) as dataFile:
        minibatches = self.makeMinibatchesList(dataFile)
        minibatchNum = 1
        accuracyRates = []
        numCorrect = 0
        #ask user how many times to run code through epoch
        #for every time user wants the code to run through epoch

        #run network
        for minibatch in minibatches:
            tOutput = minibatch[9]
            if tOutput == 'positive':
                tOutput = 1.0
            elif tOutput == 'negative':
                tOutput = 0.0

            minibatchInputs = minibatch[0:9]
            inputs = np.reshape(
                (np.asarray(minibatchInputs)),
                (self.layerSizes[0], 1)
            )

            #This section runs the actual network
            expOutput = self.feedforward(inputs)
            self.updateWB(expOutput, inputs)

            #This section evaluates the efficiency
            expOutput = round(expOutput)

            if expOutput == tOutput:
                numCorrect = numCorrect + 1
            groupsOf = 50
            if minibatchNum % groupsOf == 0:
                percentCorrectStr = str(
                    round((numCorrect/groupsOf)*100)
                ) + str(" %")
                accuracyRates.append(percentCorrectStr)
                numCorrect = 0
            minibatchNum = minibatchNum + 1

            #this is where I have the code loop through the epoch again
            
        print(f"Accuracy rates in batches of {groupsOf}: {accuracyRates}")

def makeMinibatchesList(self, dataFile):
    minibatches = []
    for minibatch in dataFile:
        minibatchSplit = minibatch.strip().split(",")
        for i in range(len(minibatchSplit)-1):
            if minibatchSplit[i] == "x":
                minibatchSplit[i] = 1.0
            else:
                minibatchSplit[i] = 0.0
        minibatches.append(minibatchSplit)
    return minibatches