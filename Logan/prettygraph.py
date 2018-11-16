import matplotlib.pyplot as plt
import math
import pylab
import numpy
import csv

with open("outputData.csv", newline='') as csvfile:
    readData = csv.reader(csvfile, delimiter=' ')
    for row in readData:
        rowStr = "".join(row)
        rowSplit = rowStr.split(",")
        experimentalOutput = rowSplit[5]
        print(experimentalOutput)

x = numpy.linspace(-20,20,100)
y = numpy.sin(x)

plt.plot(x,y)
plt.axis([-20,20,-5,5])
plt.show()