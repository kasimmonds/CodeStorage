#load modules
import xlrd
import numpy as np
import matplotlib.pyplot as plt


# Open the workbook
xl_workbook = xlrd.open_workbook('/Users/kasimmonds/Desktop/gps.xlsx')

#select sheet
sheet = xl_workbook.sheet_by_index(1)

#define empty array with rows, columns
data = np.zeros((121,2))

#for each column
for j in range(0,2):
    #for each row (start with 1 to disclude labels)
   for i in range(1,122):
       data[i,j] = sheet.cell_value(i,j)
       print data[i,j]

#plot figure
plt.figure(1)

#plot column 1 against column 2
plt.plot(data[:,0], data[:,1])
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Title')
plt.grid(True)
plt.show()
