from PIL import Image
import numpy as np
import glob


def main():
    with open("new_natural_images.csv", "r+") as dataFile:
        for fileName in glob.glob('natural_images_stretched//*.jpg'):
            inputs = np.asarray(Image.open(fileName))

            # need intensities array, not bins array
            inputs = np.ndarray.tolist(inputs[0])
            # inputs = str(inputs).strip("[]")
            inputs = str(inputs)
            fileName = fileName.strip("natural_images_stretched'\\'")
            dataFile.write(f"{fileName}, {inputs}\n")


if __name__ == "__main__":
    main()