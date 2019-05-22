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
    with open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\mixed_new_natural_images.csv", newline=''
              ) as dataFile:
        for row in dataFile:
            minibatch = []
            dataSplit = row.strip().split(";")
            jpgName = dataSplit[0]
            if jpgName[0:5] == "plane":
                tOut = 1.0
            else:
                tOut = 0.0
            pixelData = dataSplit[1]
            pixelData = pixelData.strip(" [[]]")
            pixelData = pixelData.split("], [")
            # print(pixelData)
            for triplet in pixelData:
                intensities = triplet.split(",")
                minibatch.append(int(intensities[0]))
           # minibatch = np.asarray(minibatch)
            inputs.append(minibatch)
            outputs.append(tOut)

    # print(inputs)
    inputs = np.asarray(inputs) #Keras docs says arrays in list but oh well
    outputs = np.asarray(outputs)
    model = Sequential()

    model.add(Dense(activation='sigmoid', input_dim=len(inputs[0]), units=1))

    # # peaks at about 0.75 around 120 epochs
    # model.add(Dense(activation='sigmoid', input_dim=len(inputs), units=70))
    # model.add(Dense(activation='sigmoid', input_dim=70, units=70))
    # model.add(Dense(activation='sigmoid', input_dim=70, units=1))

    model.compile(optimizer='SGD',
                  loss='mean_squared_error',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(inputs, outputs, epochs=170, batch_size=1)

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
