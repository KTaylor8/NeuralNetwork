from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics, utils
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# doesn't actually work; it misses the 10.5% of photos that are airplanes because there are too many non-airplanes in the data and it just says everything is not an airplane


def main():
    inputs = []
    outputs = []
    #C:\Users\s-2508690\Desktop\NeuralNetwork
    with open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\naturalImagesBinary.csv", newline=''
              ) as dataFile:
        for row in dataFile:
            dataSplit = row.strip().split(",")
            if dataSplit[0][0:5] == "plane":
                tOut = 1.0
            else:
                tOut = 0.0
            intensities = dataSplit[1:11]
            inputs.append(intensities)
            outputs.append(tOut)
    data = np.asarray(inputs)
    labels = np.asarray(outputs)

    model = Sequential()

    # peaks at an average of 0.76 around 120-140 epochs, got to 0.75 ~25 epochs
    model.add(Dense(activation='sigmoid', input_dim=10, units=70))
    model.add(Dense(activation='sigmoid', input_dim=70, units=80))
    model.add(Dense(activation='sigmoid', input_dim=80, units=80))
    model.add(Dense(activation='sigmoid', input_dim=80, units=1))

    model.compile(optimizer='SGD',
                  loss='mean_squared_error',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(data, labels, epochs=170, batch_size=1)

    model.summary()

    # Plot accuracy rates
    plt.plot(np.asarray(history.history["binary_accuracy"])*100)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    axes = plt.gca()
    axes.set_ylim(40, 100)
    plt.show()


if __name__ == "__main__":
    main()
