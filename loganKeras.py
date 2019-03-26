#Katie's most recent code

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics
import keras.utils
import csv
import numpy as np

model = Sequential()  # sets up network to be linear stack of layers

# adds 32 layers with the activation function relu and 5 inputs

model.add(Dense(1, activation='relu', input_dim=60))
# adds 10 layers with the activation function softmax
# model.add(Dense(10, activation='softmax'))
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
        inputs.append(float(i) for i in dataSplit[0:60])
        if dataSplit[60] == 'M':  # theoretical output
            outputs.append(1.0)
        elif dataSplit[60] == 'R':
            outputs.append(0.0)
        outputs.append(dataSplit[60])
data = np.asarray((inputs))
labels = np.asarray((outputs))
model.fit(data, labels, epochs=20, batch_size=10)

model.summary()
