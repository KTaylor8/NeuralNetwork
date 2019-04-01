from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics
import keras.utils
import csv
import numpy as np
import matplotlib.pyplot as plt

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
        for i in range(len(dataSplit)-1):
            dataSplit[i] = float(dataSplit[i])
            if dataSplit[-1] == 'M': # theoretical output
                dataSplit[-1] = 1.0
            elif dataSplit[-1] == 'R':
                dataSplit[-1] = 0.0
        inputs.append(dataSplit[0:60])
        outputs.append(dataSplit[60])
data = np.asarray((inputs))
labels = np.asarray((outputs))
history = model.fit(data, labels, epochs=50, batch_size=10)

print(history.history.keys())
model.summary()

axes = plt.gca()
plt.plot((np.asarray(history.history['binary_accuracy']))*100)
plt.title('Model accuracy') 
axes.set_ylim([40,100])
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()