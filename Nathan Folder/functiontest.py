import numpy as np
import math
import random

def main():
    miniBatch = np.array([i] for i in input("Enter number"))
    print(miniBatch)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()