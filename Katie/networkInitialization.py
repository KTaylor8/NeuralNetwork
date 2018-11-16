# enter into terminal:
# import network
# net = network.Network([784, 30, 10])

import csv
import os
import random as rdm
import numpy as np


def main():
    class Network():
        with open("ticTacToeData.csv", newline='') as csvfile:
            readData = csv.reader(csvfile, delimiter=' ')
            iterationAccuracyDict = {}
            for row in readData:  # each row is a single element list
                rowStr = "".join(row)
                rowSplit = rowStr.split(",")
                print(rowSplit)
                theoreticalOutput = rowSplit[9]
                rowInputs = rowSplit[0:8]
                print(rowInputs)
                break

                # append theoretical matches exp yes or no
                # iterationAccuracyDict.append("")

    # csv python file handling
    # split()
    # list of tuples w/ each row's data

    # # inputs file into python dictionary and converts it into list
    #     file = ('C:/Users/s-2508690/Desktop/NeuralNetwork/ticTacToeData.xlsx')
    #     x1 = pd.ExcelFile(file)
    #     df1 = x1.parse('Sheet1')  # con

    #     df1 = df1.values()  # converts into an array
    #     while row


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
