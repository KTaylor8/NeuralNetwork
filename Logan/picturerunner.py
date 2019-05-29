from numpy import loadtxt
from keras.models import load_model
from itertools import chain
 
model = load_model('output//model.h5')
model.summary()
inputs = Image.open("output//demo1.jpg")
inputs = list(chain.from_iterable(inputs))
planeorperson = str(input("Is this a plane or a person?"))
if str.lower(planeorperson) == "plane":
    tOut = 1.0
else:
    tOut = 0.0
print(inputs)

# basically i now need to turn the inputs into a readable list, i removed the garbage data by hand because i cant program and now I need it do be done by a computer. i think the best bet is to save the array to a csv and then read it from the csv as a list, because it wont let you convert it directly to a list. saving the model to a file might not work either but thats fine; im learning https://machinelearningmastery.com/save-load-keras-deep-learning-models/
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, tOut, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))