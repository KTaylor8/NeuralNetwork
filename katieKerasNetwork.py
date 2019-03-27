from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics, utils
import csv
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()  # sets up network to be linear stack of layers

model.add(Dense(1, activation='relu', input_dim=60))
# adds a layer with the activation function relu, 60 inputs, and 1 output

# .compile() initializes network type
model.compile(optimizer='SGD',  # this is the type of backprop
              loss='mean_squared_error',  # this is the type of cost function
              metrics=[metrics.binary_accuracy])  # this is the fitness function to determine success

inputs = []
outputs = []
#C:\Users\s-2508690\Desktop\NeuralNetwork
with open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\sonar.all-data.csv", newline=''
          ) as dataFile:
    for row in dataFile:
        dataSplit = row.strip().split(",")
        for i in range(len(dataSplit)-1):
            dataSplit[i] = float(dataSplit[i])
            if dataSplit[-1] == 'M':  # theoretical output
                dataSplit[-1] = 1.0
            elif dataSplit[-1] == 'R':
                dataSplit[-1] = 0.0
        inputs.append(dataSplit[0:60])
        outputs.append(dataSplit[60])
data = np.asarray((inputs))
labels = np.asarray((outputs))

# model.fit(training data, experimental output, batch_size=num samples per gradient update)

history = model.fit(data, labels, validation_split=0.25,
                    epochs=50, batch_size=16, verbose=1)

model.summary()

# Plot training & validation accuracy values
plt.plot(history.history[metrics.binary_accuracy])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
