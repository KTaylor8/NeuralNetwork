from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers, metrics, utils
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

inputs = []
outputs = []
with open(r"planesandpeoplerandom.csv", newline=''
          ) as dataFile:
    for row in dataFile:
        dataSplit = row.strip().split(",")
        if dataSplit[0][0:5] == "plane":
            tOut = 1.0
        else:
            tOut = 0.0
        intensities = dataSplit[1:101]
        inputs.append(intensities)
        outputs.append(tOut)
data = np.asarray(inputs)
labels = np.asarray(outputs)

model = Sequential()

model.add(Dense(activation='sigmoid', input_dim=100, units=100))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(activation='sigmoid', input_dim=100, units=1))

model.compile(optimizer='Adadelta',
                loss='logcosh',
                metrics=[metrics.binary_accuracy])

history = model.fit(data, labels, epochs=10, batch_size=1)

model.summary()
model.save("model.h5")

plt.plot(np.asarray(history.history["binary_accuracy"])*100)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
axes = plt.gca()
axes.set_ylim(40, 100)
plt.show()