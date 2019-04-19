from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics, utils
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    inputs = []
    outputs = []
    with open(r"natural_images.csv", newline=''
              ) as dataFile:
        for row in dataFile:
            dataSplit = row.strip().split(",")
            if dataSplit[0][0:5] == "plane":
                tOut = 1.0
            else:
                tOut = 0.0
            intensities = dataSplit[1].split(" ")
            inputs.append(intensities)
            outputs.append(tOut)
    data = np.asarray((inputs))
    labels = np.asarray((outputs))

    # model = Sequential()  # sets up network to be linear stack of layers

    # model.add(Dense(1, activation='tanh', input_dim=60))
    # # adds a layer with the activation function relu, 60 inputs, and 1 output

    # # .compile() initializes network type
    # model.compile(optimizer='SGD',  # this is the type of backprop
    #               loss='mean_squared_error',  # this is the type of cost function
    #               metrics=[metrics.binary_accuracy])  # this is the fitness function to determine success

    # # model.fit(training data, experimental output, batch_size=num samples per gradient update (lower num means learns more quickly initially but doesn't increase maximum accuracy))
    # history = model.fit(data, labels, epochs=5000, batch_size=1)

    # model.summary()

    # # Plot accuracy rates
    # plt.plot(np.asarray(history.history["binary_accuracy"])*100)
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # axes = plt.gca()
    # axes.set_ylim(40, 100)
    # plt.show()


if __name__ == "__main__":
    main()
