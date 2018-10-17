import numpy as np
import math
import os
import random
import time

#Identifies inputs
f = np.array([float(x) for x in input().split()])
print (f)
l = f.size
print (l)

#creates the weight and biases
w = np.random.randint(0, high=3, size=(1, l))
print(w)
b = -5

#finds the sum of the weights*inputs, adds bias
f1 = np.multiply(f, w)
print(f1)
time.sleep(2)

f2 = np.sum(f1, axis=None) + b
print(f2)
time.sleep(2)

#places sum into a sigmoid function
f3 = 1/(1+(math.e**f2))
print(f3)

#creates threshold to find output value (0 or 1)
if f3 < 0.5:
    print('1')
elif f3 >= 0.5:
    print('0')