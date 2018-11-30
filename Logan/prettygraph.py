import matplotlib.pyplot as plt
import numpy
import csv

#define empty list to add numbers from csv to
listExperimentals1 = []
listExperimentals2 = []
listExperimentals3 = []

#add all numbers from csv file to list
with open("outputData.csv", newline='') as csvfile:
    readData = csv.reader(csvfile, delimiter=' ')
    for row in readData:
        rowStr = "".join(row)
        rowSplit = rowStr.split(",")
        #convert to float and turn into percentage
        experimentalOutput1 = float(rowSplit[5])*100
        experimentalOutput2 = float(rowSplit[9])*100
        experimentalOutput3 = float(rowSplit[13])*100
        listExperimentals1.append(experimentalOutput1)
        listExperimentals2.append(experimentalOutput2)
        listExperimentals3.append(experimentalOutput3)

#define x and y values
#x is numbers between 0 and 958, with 958 of them
x = numpy.linspace(0,958,958)
#y is previous list, all numbers between 0 and 100
y1 = listExperimentals1
y2 = listExperimentals2
y3 = listExperimentals3

#plot the data on the graph
plt.plot(x,y1,"r-",label="Test run 1")
plt.plot(x,y2,"b-",label="Test run 2")
plt.plot(x,y3,"y-",label="Test run 3")

#change axis labels
plt.xlabel("Iteration")
plt.ylabel("Percentage Correct")
plt.title("Percentage Correct over Time")
plt.axis([0,958,0,100])

#show graph
plt.show()