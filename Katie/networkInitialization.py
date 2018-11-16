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

                # append theoretical matches exp yes or no:
                # iterationAccuracyDict.append("")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
