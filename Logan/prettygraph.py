import matplotlib.pyplot as plt
import math
import pylab
import numpy
import csv

#define empty list to add numbers from csv to
listExperimentals = []

#add all numbers from csv file to list
with open("outputData.csv", newline='') as csvfile:
    readData = csv.reader(csvfile, delimiter=' ')
    for row in readData:
        rowStr = "".join(row)
        rowSplit = rowStr.split(",")
        #convert to float and turn into percentage
        experimentalOutput = float(rowSplit[5])*100
        listExperimentals.append(experimentalOutput)

#define x and y values
#x is numbers between 0 and 958, with 958 of them
x = numpy.linspace(0,958,958)
#y is previous list, all numbers between 0 and 100
y = listExperimentals

#plot the data on the graph
plt.plot(x,y)

#change axis labels
plt.xlabel("Iteration")
plt.ylabel("Percentage Correct")
plt.title("Percentage Correct over Time")
plt.axis([0,958,0,100])

#show graph
plt.show()