from PIL import Image
import numpy as np


def main():
    inputs = []
    try:
        # Relative Path
        img = Image.open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\flower_0054.jpg")
        
        # Getting histogram of image
        print(img.histogram())
        inputs = img.histogram()

    except IOError:
        pass

    inputs = np.asarray(inputs)
    print(inputs)

if __name__ == "__main__":
    main()
