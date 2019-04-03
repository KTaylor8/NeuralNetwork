from PIL import Image
import glob

def main():
    image_list = []
    for filename in glob.glob('natural_images_grayscale/*.jpg'):
            img = Image.open(filename)
            im = img.histogram()
            print(im, "1")
            image_list.append(img)
            
    print(image_list)

if __name__ == "__main__":
    main()