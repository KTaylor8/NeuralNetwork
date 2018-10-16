import numpy as np
import math
import os
import random
import time as t

#Identifies inputs
i1 = int((input("Input 1 (0,1) ")))
i2 = int((input("Input 2 (0,1) ")))
i3 = int((input("Input 3 (0,1) ")))

#Places them into an array
i = np.array([i1, i2, i3])
print(i)

#creates the weight and biases
w = np.random.randint(0, high=2, size=(1, 3))
print(w)
b = -1

#finds the sum of the weights*inputs, adds bias
f1 = np.multiply(i, w)
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