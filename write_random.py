import xlrd
from xlutils.copy import copy as xl_copy
import random


name = 'excel/final.xls'
rworkbook = xlrd.open_workbook(name, formatting_info=True)
wworkbook = xl_copy(rworkbook)
sheet = rworkbook.sheet_by_name("phi")
wsheet = wworkbook.add_sheet("phi3")
for i in range(200):
    if i < 100:
        e = 0.02/(2**(i/50.0))
    else:
        e = 0.02/(2**(100/50.0))
    for j in range(4):
        value = sheet.cell(i+1, j+1).value + random.uniform(-e, e)
        wsheet.write(i+1, j+1, value)
wworkbook.save(name)