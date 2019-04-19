from PIL import Image
import numpy as np
import glob
import os
import csv
def main():
    os.getcwd()
    os.chdir(r'C:\Users\s-2508690\Desktop\NeuralNetwork')
    image_list = []
    print(glob.glob('natural_images_stretched/*.jpg'))
    for filename in glob.glob('natural_images_stretched/*.jpg'):
        inputsList.append(filename)
    with open(r'C:\Users\s-2508690\Desktop\NeuralNetwork\imageData.csv'):
        writer = csv.writer(r'C:\Users\s-2508690\Desktop\NeuralNetwork\imageData.csv')
        writer.writerows(inputsList)
    print(inputsList)
    #img = Image.open(filename)
    #im = img.histogram()
    #print(im, "1")

if __name__ == "__main__":
    main()