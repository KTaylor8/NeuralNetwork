import numpy as np
import math
import os
import random
import time

#define sigmoid function
def sigmoid(x):
    f3 = 1/(1+(math.e**((-1)*x)))
    return f3

#define sum of dot product function
def __sumofdot(x, w, b):
    f1 = np.multiply(x, w)
    f2 = np.sum(f1, axis=None) + b
    return f2

#define main code
def main():
    #Create list
    f = np.array([float(x) for x in input().split()])
    l = f.size

    #assign weights and bias
    w = np.random.randint(0, high=3, size=(1,l))
    b = -5

    #find sum of dot product of list
    f2 =__sumofdot(f, w, b)
    
    #apply activation function
    f3 = sigmoid(f2)

    #set threshold and decide to fire or not
    if f3 > 0.5:
        print('1')
    if f3 <= 0.5:
        print('0')

#run main code
main()