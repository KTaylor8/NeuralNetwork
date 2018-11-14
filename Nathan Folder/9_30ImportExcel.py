# this program will open, edit, and display a spreadsheet from excel

# import os
import os

# import Numpy library
import numpy as np

# import pandas library
import pandas as pd

<<<<<<< HEAD
# defines file
file = ('C:/Users/s-2508690/Desktop/Python/FBD Program Learning/Pythontest.xlsx')
=======
#defines file
file = ( 'C:\Users\s-2508690\Desktop\NeuralNetwork\Nathan Folder\doil.xlsx' )
>>>>>>> 4bb19e026f0b2c0b6b90567c93c01f2c6f94887a

# puts it into python
x1 = pd.ExcelFile(file)

print(x1.sheet_names)

# places excel program into database
df1 = x1.parse('Sheet1')

# converts database into an array
df1 = df1.values
print(df1)
print(df1.shape)

<<<<<<< HEAD
df2 = np.array(np.random.random((3, 2)))
print(df2)

df3 = np.vstack((df1, df2))

print(df3)
=======


>>>>>>> 4bb19e026f0b2c0b6b90567c93c01f2c6f94887a
