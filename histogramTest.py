from PIL import Image
import numpy as np
import glob


def main():
    inputsList = []
    for fileName in glob.glob('natural_images_stretched\\*.jpg'):
        inputs = np.histogram(Image.open(fileName))
        inputsList.append((fileName, inputs))
    print(inputsList[0])


if __name__ == "__main__":
    main()


# def main():
#     inputs = []
#     try:
#         # Relative Path
#         img = Image.open(r"C:\Users\s-2508690\Desktop\NeuralNetwork\flower_0054.jpg")

#         # Getting histogram of image
#         print(img.histogram())
#         inputs = img.histogram()

#     except IOError:
#         pass

#     inputs = np.asarray(inputs)
#     print(inputs)

# if __name__ == "__main__":
#     main()
