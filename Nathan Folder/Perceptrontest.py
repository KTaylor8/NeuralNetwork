import numpy as np
import math
import os
import random
import time as t


#Identifies inputs
f = np.array([int(x) for x in input().split()])
print (f)
l = f.size
print (l)

#creates the weight and biases
w = np.random.randint(0, high=2, size=(1, l))
print(w)
b = -1

#finds the sum of the weights*inputs, adds bias
f1 = np.multiply(f, w)
print(f1)
t.sleep(2)

f2 = np.sum(f1, axis=None) + b
print(f2)
t.sleep(2)

#threshold for activation (step function)
if f2 > 0:
    print('1')
elif f2 <= 0:
    print('0')