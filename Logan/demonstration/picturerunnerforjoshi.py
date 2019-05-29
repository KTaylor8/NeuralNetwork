from numpy import loadtxt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers, metrics, utils
from itertools import chain
from PIL import Image
import csv
import numpy as np
import glob

model = load_model('model.h5')
model.summary()
# fileName = input("What is the file name?")
# inputs = np.array(Image.open(str(f"output\\{fileName}.jpg")))
# inputs = list(chain.from_iterable(inputs))

# planeorperson = str(input("Is this a plane or a person?"))
# if str.lower(planeorperson) == "plane":
#     tOut = 1.0
# else:
#     tOut = 0.0
tOut1 = 0.0
with open("dataStorage.csv", "w") as dataStorage:
    for fileName in glob.glob('morepeople//*.jpg'):
        inputss = np.array(Image.open(fileName))
        inputss = list(chain.from_iterable(inputss))
        dataStorage.write(f"{tOut1}, {inputss}\n")

tOut = []
inputs = []
with open(r"dataStorage.csv", newline=''
          ) as dataFile:
    for row in dataFile:
        intensities = []
        dataSplit = row.strip().split(",")
        for x in dataSplit[2:401]:
            intensities.append(x)
        intensities = intensities[::4]
        intensities = [float(x) for x in intensities]
        inputs.append(intensities)
        tOut.append(0.0)
data = np.asarray(inputs)

score = model.evaluate((data), np.expand_dims(np.array(tOut),1), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))