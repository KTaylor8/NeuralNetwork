from PIL import Image
import numpy as np
import glob

def main():
    image_list = []
    for filename in glob.glob('C:\Users\s-2513816\Desktop\NeuralNetwork\natural_images_stretched/*.jpg'):
        image_list.append(filename)
    print(image_list)

if __name__ == "__main__":
    main()