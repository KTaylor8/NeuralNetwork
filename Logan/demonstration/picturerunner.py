from numpy import loadtxt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers, metrics, utils
from itertools import chain
from PIL import Image
import csv
import numpy as np

model = load_model('model.h5')
model.summary()
fileName = input("What is the file name? ")
inputs = np.array(Image.open(f"output\\{fileName}.jpg"))
inputs = list(chain.from_iterable(inputs))

planeorperson = input("Is this a plane or a person? ")
if planeorperson.lower() == "plane":
    tOut = 1.0
else:
    tOut = 0.0

with open("dataStorage.csv", "w") as dataStorage:
    dataStorage.write(f"{tOut}, {inputs}\n")

with open(r"dataStorage.csv", newline=''
          ) as dataFile:
    for row in dataFile:
        dataSplit = row.strip().split(",")
        intensities = []
        for x in dataSplit[2:401]:
            intensities.append(x)
        intensities = intensities[::4]
        intensities = [float(x) for x in intensities]
        inputs.append(intensities)
data = np.asarray(inputs)
data = np.reshape((np.asarray(data[-1])), (1, 100))

score = model.evaluate((data), np.expand_dims(np.array(tOut), 1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
if score[1] == 1:
    if planeorperson == "plane":
        print("This is a plane!")
    elif planeorperson == "person":
        print("This is a person!")
elif score[1] == 0:
    if planeorperson == "plane":
        print("This is a person!")
    elif planeorperson == "person":
        print("This is a plane!")
