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
    with open(r"new_natural_images.csv", newline=''
              ) as dataFile:
        for row in dataFile:
            dataSplit = row.strip().split(";")
            jpgName = dataSplit[0]
            if jpgName[0:5] == "plane":
                tOut = 1.0
            else:
                tOut = 0.0
            pixelData = dataSplit[1]
            pixelData = pixelData.strip(" [[]]")
            pixelData = pixelData.split("], [")
            for triplet in pixelData:
                intensities = triplet.split(",")
                inputs.append(int(intensities[0]))
            outputs.append(tOut)
    # data = (inputs)
    # ValueError: Input arrays should have the same number of samples as target arrays. Found 1 input samples and 6899 target samples.
    data = np.reshape(
        (inputs),
        (1, len(inputs))
    )
    print(data)
    labels = outputs

    model = Sequential()

    # peaks at about 0.75 around 120 epochs
    model.add(Dense(activation='sigmoid', input_dim=len(data), units=70))
    model.add(Dense(activation='sigmoid', input_dim=70, units=70))
    model.add(Dense(activation='sigmoid', input_dim=70, units=1))

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
