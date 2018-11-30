import matplotlib.pyplot as plt
import numpy
import csv

#define empty lists to add numbers from csv to
listExperimentals1 = []
listExperimentals2 = []
listExperimentals3 = []

#define empty lists for percentages
listPercentage1 = []
listPercentage2 = []
listPercentage3 = []

#importing all data
with open("newOutputData.csv", newline='') as csvfile:
    readData = csv.reader(csvfile, delimiter=' ')
    for row in readData:
        rowStr = "".join(row)
        rowSplit = rowStr.split(",")
        actualValue = int(rowSplit[2])
        experimentalValue1 = int(rowSplit[3])
        experimentalValue2 = int(rowSplit[4])
        experimentalValue3 = int(rowSplit[5])
        
        #comparing experimental data to actual data and adding result to list of experimental data
        if experimentalValue1 == actualValue:
            listExperimentals1.append(1)
        else:
            listExperimentals1.append(0)
        if experimentalValue2 == actualValue:
            listExperimentals2.append(1)
        else:
            listExperimentals2.append(0)
        if experimentalValue3 == actualValue:
            listExperimentals3.append(1)
        else:
            listExperimentals3.append(0)
        
        #summing current total of correct then dividing by number of guesses to get total percentage as a decimal
        listPercentage1.append((sum(listExperimentals1))/(int(rowSplit[0])))
        listPercentage2.append((sum(listExperimentals2))/(int(rowSplit[0])))
        listPercentage3.append((sum(listExperimentals3))/(int(rowSplit[0])))

#converting decimals to percentage scale
listPercentage1 = [i*100 for i in listPercentage1]
listPercentage2 = [i*100 for i in listPercentage2]
listPercentage3 = [i*100 for i in listPercentage3]

#define x and y values
#x is numbers between 0 and 958, with 958 of them
x = numpy.linspace(0,957,958)
#y is previous list, all numbers between 0 and 1
y1 = listPercentage1
y2 = listPercentage2
y3 = listPercentage3

#plot the data on the graph
plt.plot(x,y1,"r-",label="Test run 1")
plt.plot(x,y2,"b-",label="Test run 2")
plt.plot(x,y3,"y-",label="Test run 3")

#change axis labels
plt.xlabel("Iteration")
plt.ylabel("Percentage Correct")
plt.title("Percentage Correct Over Time")
plt.axis([0,958,0,100])

#show graph
plt.show()