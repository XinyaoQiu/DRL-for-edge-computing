
import matplotlib.pyplot as plt

import xlrd
from xlutils.copy import copy as xl_copy

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0

rworkbook = xlrd.open_workbook("excel/final.xls")
sheet = rworkbook.sheet_by_name("energy")
episodes = [sheet.cell(i+1, 0).value for i in range(200)]
phi_1 = [sheet.cell(i+1, 1).value for i in range(200)]
phi_2 = [sheet.cell(i+1, 2).value for i in range(200)]
phi_3 = [sheet.cell(i+1, 3).value for i in range(200)]
phi_4 = [sheet.cell(i+1, 4).value for i in range(200)]
phi_avg1 = [sheet.cell(201, 1).value for i in range(200)]
phi_avg2 = [sheet.cell(201, 2).value for i in range(200)]
phi_avg3 = [sheet.cell(201, 3).value for i in range(200)]
phi_avg4 = [sheet.cell(201, 4).value for i in range(200)]
plt.plot(episodes, phi_1)
plt.plot(episodes, phi_2)
plt.plot(episodes, phi_3)
plt.plot(episodes, phi_4)
plt.plot(episodes, phi_avg1)
plt.plot(episodes, phi_avg2)
plt.plot(episodes, phi_avg3)
plt.plot(episodes, phi_avg4)


plt.savefig("output/energy_vs_ddl")
plt.close()

rworkbook = xlrd.open_workbook("excel/final.xls")
sheet = rworkbook.sheet_by_name("phi")
episodes = [sheet.cell(i+1, 0).value for i in range(200)]
phi_1 = [sheet.cell(i+1, 1).value for i in range(200)]
phi_2 = [sheet.cell(i+1, 2).value for i in range(200)]
phi_3 = [sheet.cell(i+1, 3).value for i in range(200)]
phi_4 = [sheet.cell(i+1, 4).value for i in range(200)]
phi_avg1 = [sheet.cell(201, 1).value for i in range(200)]
phi_avg2 = [sheet.cell(201, 2).value for i in range(200)]
phi_avg3 = [sheet.cell(201, 3).value for i in range(200)]
phi_avg4 = [sheet.cell(201, 4).value for i in range(200)]
plt.plot(episodes, phi_1)
plt.plot(episodes, phi_2)
plt.plot(episodes, phi_3)
plt.plot(episodes, phi_4)
plt.plot(episodes, phi_avg1)
plt.plot(episodes, phi_avg2)
plt.plot(episodes, phi_avg3)
plt.plot(episodes, phi_avg4)


plt.savefig("output/phi_vs_ddl")