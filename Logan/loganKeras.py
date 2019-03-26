from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
import csv
import numpy as np

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.asarray(())
with open(r"powerPlantData.csv", newline=''
    ) as dataFile:
        for row in dataFile:
            dataSplit = row.strip().split(",")
        data.append(dataSplit)
    
labels = np.random.randint(10, size=(1000, 1))

one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

model.fit(data, one_hot_labels, epochs=10, batch_size=32)

model.summary()