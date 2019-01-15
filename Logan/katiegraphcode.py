import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
import csv


def graphInit():
    redPatch = mpatches.Patch(color='red', label='Test Run 1')
    plt.legend(handles=[redPatch], loc="upper right")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage Correct")
    plt.title("Percentage Correct Over Time")
    plt.axis([0, 1000, 0, 100])
    return graphLine,


def graphUpdate(frame):
    xdata.append(frame)
    ydata.append(y1[int(frame)])
    graphLine.set_data(xdata, ydata)
    return graphLine,


# define empty lists to add numbers from csv to
listExperimentals1 = []


# define empty lists for percentages
listPercentage1 = []

# importing all data
with open("newOutputData.csv", newline='') as csvfile:
    readData = csv.reader(csvfile, delimiter=' ')
    for row in readData:
        rowStr = "".join(row)
        rowSplit = rowStr.split(",")
        actualValue = int(rowSplit[2])
        experimentalValue1 = int(rowSplit[3])

        # comparing experimental data to actual data and adding result to list of experimental data
        if experimentalValue1 == actualValue:
            listExperimentals1.append(1)
        else:
            listExperimentals1.append(0)

        # summing current total of correct then dividing by number of guesses to get total percentage as a decimal
        listPercentage1.append((sum(listExperimentals1))/(int(rowSplit[0])))


# converting decimals to percentage scale
listPercentage1 = [i*100 for i in listPercentage1]


# define x and y values
# #x is numbers between 0 and 958, with 958 of them
# x = np.linspace(0,957,958)
# #y is previous list, all numbers between 0 and 1
plt.switch_backend('TkAgg')
y1 = listPercentage1


fig, ax = plt.subplots()
xdata, ydata = [], []
graphLine, = plt.plot([], [], 'r-', animated=True)

ani = animation.FuncAnimation(fig,
                              graphUpdate,
                              frames=np.linspace(1, 958, 479),
                              init_func=graphInit,
                              blit=True)
plt.show()
