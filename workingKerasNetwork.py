from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics, utils
import csv
import numpy as np

model = Sequential()  # sets up network to be linear stack of layers

model.add(Dense(1, activation='relu', input_dim=60))
# adds a layer with the activation function relu, 60 inputs, and 1 output

# .compile() initializes network type
model.compile(optimizer='SGD',  # this is the type of backprop
              loss='mean_squared_error',  # this is the type of cost function
              metrics=[metrics.binary_accuracy])  # this is the fitness function to determine success

inputs = []
outputs = []
with open(r"sonar.all-data.csv", newline=''
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
model.fit(data, labels, epochs=200, batch_size=20)

model.summary()
