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

# labels = np.random.randint(10, size=(9000, 568)) #weights array
# labels = np.random.uniform(low=-1, high=1, size=(9568, 9568))


# model.add(Flatten())  # reduces shape of labels (tuples)


# outputs an array containing binary arrays (rows) that denote the presence (1.) or absence (0.) of a class (separate types of nums) in each index of the input array. # classes = # columns
# Ex: # Inputting w/ an array of 5 labels out of a set of 3 classes {0, 1, 2}:
# array([0, 2, 1, 2, 0])
# output:
# array([[ 1.,  0.,  0.],
#        [ 0.,  0.,  1.],
#        [ 0.,  1.,  0.],
#        [ 0.,  0.,  1.],
#        [ 1.,  0.,  0.]], dtype=float32)

# hot_labels = keras.utils.to_categorical(labels)
# hot_labels = np.reshape(hot_labels, (9000, 568*1))

# model.fit(training data, experimental output, batch_size=num samples per gradient update)
model.fit(data, labels, epochs=50, batch_size=20)

model.summary()
