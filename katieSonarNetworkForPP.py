from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers, metrics, utils
import csv
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(1, activation='tanh', input_dim=60))

model.compile(optimizer='SGD',
              loss='mean_squared_error',
              metrics=[metrics.binary_accuracy])

inputs = []
outputs = []
with open(r"sonar.all-data.csv", newline=''
          ) as dataFile:
    for row in dataFile:
        dataSplit = row.strip().split(",")
        for i in range(len(dataSplit)-1):
            dataSplit[i] = float(dataSplit[i])
            if dataSplit[-1] == 'M':
                dataSplit[-1] = 1.0
            elif dataSplit[-1] == 'R':
                dataSplit[-1] = 0.0
        inputs.append(dataSplit[0:60])
        outputs.append(dataSplit[60])
data = np.asarray((inputs))
labels = np.asarray((outputs))

history = model.fit(data, labels, epochs=200, batch_size=1)

model.summary()

# Plot accuracy rates
plt.plot(np.asarray(history.history["binary_accuracy"])*100)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
axes = plt.gca()
axes.set_ylim(40, 100)
plt.show()

# # Plot loss (cost) values
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()
