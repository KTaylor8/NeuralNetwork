#This program will import a text file and display it on the screen
import os
print(os.getcwd())
os.listdir('.')
file = open('C:/Users/s-2508690/Desktop/Python/FBD Program Learning/9_30testtext.txt', 'r+')
content = file.read()
print(content)
filewrite = input("Add extra text into file: ")

file.write(filewrite)
file.close()