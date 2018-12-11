import os
import numpy as np
import random
import time

def createList(x):
    while True:
        if x == "integer":
            listx = np.random.randint(0, 1, [3, 3], dtype=1)
            break
        elif x == "float":
            listx = np.random.randn()
            break
        else:
            print("Incorrect, try again")
    return listx

def main():
    x = input("What list do you want, integer or float? ")
    print("The list is:", createList(x))

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()     