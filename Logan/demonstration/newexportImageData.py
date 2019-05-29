from PIL import Image
import numpy as np
import glob
from itertools import chain


def main():
    with open("planesandpeoplenew.csv", "r+") as dataFile:
        for fileName in glob.glob('output//*.jpg'):
            inputs = np.array(Image.open(fileName))
            inputs = list(chain.from_iterable(inputs))
            if fileName[7:15] == 'airplane':
                name = "plane"
            else:
                name = "flower"
            dataFile.write(f"{name}, {inputs}\n")

if __name__ == "__main__":
    main()