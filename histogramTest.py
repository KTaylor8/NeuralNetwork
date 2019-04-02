from PIL import Image


def main():
    try:
        # Relative Path
        img = Image.open("flower_0054.jpg")

        # Getting histogram of image
        print(img.histogram())

    except IOError:
        pass


if __name__ == "__main__":
    main()
