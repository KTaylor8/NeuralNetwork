# this program will open, edit, and display a spreadsheet from excel

# import os
import os

# import Numpy library
import numpy as np

# import pandas library
import pandas as pd

#defines file
file = ( 'C:\Users\s-2508690\Desktop\NeuralNetwork\Nathan Folder\doil.xlsx' )

# puts it into python
x1 = pd.ExcelFile(file)

print(x1.sheet_names)

# places excel program into database
df1 = x1.parse('Sheet1')

# converts database into an array
df1 = df1.values
print(df1)
print(df1.shape)



