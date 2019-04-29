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
    with open(r"naturalImagesBinary.csv", newline=''
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

    # peaks at about 0.75 around 120 epochs
    model.add(Dense(activation='sigmoid', input_dim=10, units=70))
    model.add(Dense(activation='sigmoid', input_dim=70, units=70))
    model.add(Dense(activation='sigmoid', input_dim=70, units=1))

    # # peaks at about 0.74 around 150 epochs
    # model.add(Dense(activation='sigmoid', input_dim=10, units=50))
    # model.add(Dense(activation='softmax', input_dim=50, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=1))

    # # peaks at about 0.730 around 110 epochs
    # model.add(Dense(activation='sigmoid', input_dim=10, units=50))
    # model.add(Dense(activation='exponential', input_dim=50, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=1))

    # # peaks at about 0.735 around 70 epochs
    # model.add(Dense(activation='sigmoid', input_dim=10, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=1))

    # # peaks at 0.71 around 90 epochs
    # model.add(Dense(activation='sigmoid', input_dim=10, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=50))
    # model.add(Dense(activation='sigmoid', input_dim=50, units=10))
    # model.add(Dense(activation='sigmoid', input_dim=10, units=1))

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
