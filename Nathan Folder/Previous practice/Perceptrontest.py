import numpy as np
import math
import os
import random
import time
"""
We start the code by defining the activation function. The activation function will map the raw output of inputs, weights, and biases onto a function, which can then be used to determine if a neuron fires or not.
"""


def sigmoid(x):
    f3 = 1/(1+(math.e**((-1)*x)))
    return f3


"""
The next step is to define the raw output function. This will take the sum of the dot product of the weights and inputs, then adds the bias.
A dot product is the product of two vectors (weights and inputs)
"""


def __sumofdot(x, w, b):
    f1 = np.multiply(x, w)
    f2 = np.sum(f1, axis=None) + b
    return f2


"""
This is the main code. This is defined as a function because it is easier to think of a neuron as a function, so it can be called any time.
"""


def main():
    """ Creates a list of inputs from the user, and defines the length of the list of inputs
    """

    f = np.array([float(x) for x in input("enter list: ").split()])
    l = f.size

    """
    Creates a list of weights for every input, and creates a set bias
    """

    w = np.random.randint(0, high=3, size=(1, l))
    print("The weights are", w)
    b = -5

    """calls the sum of the dot product and the activation function, which uses the list of inputs, weights, and bias to find the raw output
    """

    f2 = __sumofdot(f, w, b)
    print("The sum of the dot product is", f2)

    f3 = sigmoid(f2)
    print("The raw output is", f3)

    """Sets a threshold for the activation function to determine if it fires or not (This may or may not be taken out, depending on how the neural network end up)
    """

    if f3 > 0.5:
        print('1')
    if f3 <= 0.5:
        print('0')


if __name__ == main():
    import doctest
    doctest.testmod()
    main()
