import xlwt
import numpy as np

numbers = np.array([1, 2, 3, 4, 5, 6, 7])

workbook = xlwt.Workbook()
sheet = workbook.add_sheet("MADDPG")
sheet.write(0, 0, "Episodes")
sheet.write(0, 1, "Reward")
for i in range(len(numbers)):
    sheet.write(i+1, 0, numbers[i]) # row, column, value
    sheet.write(i+1, 1, numbers[i])
workbook.save('Excel_drl.xls')

    