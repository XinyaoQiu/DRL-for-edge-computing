import matplotlib.pyplot as plt
import xlrd
MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0

rworkbook = xlrd.open_workbook("excel/final.xls")
agents = [1, 2, 3, 4, 5, 6]
sheet_ddpg = rworkbook.sheet_by_name("ddpg_agent")
episodes_avg_ddpg = [sheet_ddpg.cell(202, i + 1).value for i in range(6)]
sheet_a2c = rworkbook.sheet_by_name("a2c_agent")
episodes_avg_a2c = [sheet_a2c.cell(202, i + 1).value for i in range(6)]
sheet_ppo = rworkbook.sheet_by_name("ppo_agent")
episodes_avg_ppo = [sheet_ppo.cell(202, i + 1).value for i in range(6)]
plt.figure()
plt.plot(agents, episodes_avg_ddpg, "*-")
plt.plot(agents, episodes_avg_a2c, "*-")
plt.plot(agents, episodes_avg_ppo, "*-")
plt.xlabel("agents")
plt.ylabel("average episodes")
plt.legend(["MADDPG", "MAA2C", "MAPPO"])
plt.savefig("graphs/episodes_avg.png")

rworkbook = xlrd.open_workbook("excel/Excel_ddpg.xls")
one_power = [40, 60, 80, 100, 120, 140]
sheet_ddpg = rworkbook.sheet_by_name("change_one_power_reward_3")
episodes_avg_ddpg_1 = [sheet_ddpg.cell(201, 4 * i + 1).value for i in range(6)]
episodes_avg_ddpg_2 = [sheet_ddpg.cell(201, 4 * i + 2).value for i in range(6)]
episodes_avg_ddpg_3 = [sheet_ddpg.cell(201, 4 * i + 3).value for i in range(6)]
episodes_avg_ddpg_4 = [sheet_ddpg.cell(201, 4 * i + 4).value for i in range(6)]
plt.figure()
plt.plot(one_power, episodes_avg_ddpg_1, "*-")
plt.plot(one_power, episodes_avg_ddpg_2, "*-")
plt.plot(one_power, episodes_avg_ddpg_3, "*-")
plt.plot(one_power, episodes_avg_ddpg_4, "*-")
plt.xlabel("change one power")
plt.ylabel("reward")
plt.legend(["user0", "user1", "user2", "user3"])
plt.savefig("graphs/change one power reward.png")

rworkbook = xlrd.open_workbook("excel/Excel_ddpg.xls")
one_power = [40, 60, 80, 100, 120, 140]
sheet_ddpg = rworkbook.sheet_by_name("change_one_power_energy_3")
episodes_avg_ddpg_1 = [sheet_ddpg.cell(201, 4 * i + 1).value for i in range(6)]
episodes_avg_ddpg_2 = [sheet_ddpg.cell(201, 4 * i + 2).value for i in range(6)]
episodes_avg_ddpg_3 = [sheet_ddpg.cell(201, 4 * i + 3).value for i in range(6)]
episodes_avg_ddpg_4 = [sheet_ddpg.cell(201, 4 * i + 4).value for i in range(6)]
plt.figure()
plt.plot(one_power, episodes_avg_ddpg_1, "*-")
plt.plot(one_power, episodes_avg_ddpg_2, "*-")
plt.plot(one_power, episodes_avg_ddpg_3, "*-")
plt.plot(one_power, episodes_avg_ddpg_4, "*-")
plt.xlabel("change one power")
plt.ylabel("reward")
plt.legend(["user0", "user1", "user2", "user3"])
plt.savefig("graphs/change one power energy.png")


rworkbook = xlrd.open_workbook("excel/Excel_ddpg.xls")
one_gamma = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
sheet_ddpg = rworkbook.sheet_by_name("change_one_gamma_r_mine_5")
episodes_avg_ddpg_1 = [sheet_ddpg.cell(201, 4 * i + 1).value for i in range(6)]
episodes_avg_ddpg_2 = [sheet_ddpg.cell(201, 4 * i + 2).value for i in range(6)]
episodes_avg_ddpg_3 = [sheet_ddpg.cell(201, 4 * i + 3).value for i in range(6)]
episodes_avg_ddpg_4 = [sheet_ddpg.cell(201, 4 * i + 4).value for i in range(6)]
plt.figure()
plt.plot(one_gamma, episodes_avg_ddpg_1, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_2, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_3, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_4, "*-")
plt.xlabel("change one gamma")
plt.ylabel("r_mine")
plt.legend(["user0", "user1", "user2", "user3"])
plt.savefig("graphs/change one gamma r_mine.png")

rworkbook = xlrd.open_workbook("excel/Excel_ddpg.xls")
one_gamma = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
sheet_ddpg = rworkbook.sheet_by_name("change_one_gamma_e_mine_5")
episodes_avg_ddpg_1 = [sheet_ddpg.cell(201, 4 * i + 1).value for i in range(6)]
episodes_avg_ddpg_2 = [sheet_ddpg.cell(201, 4 * i + 2).value for i in range(6)]
episodes_avg_ddpg_3 = [sheet_ddpg.cell(201, 4 * i + 3).value for i in range(6)]
episodes_avg_ddpg_4 = [sheet_ddpg.cell(201, 4 * i + 4).value for i in range(6)]
plt.figure()
plt.plot(one_gamma, episodes_avg_ddpg_1, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_2, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_3, "*-")
plt.plot(one_gamma, episodes_avg_ddpg_4, "*-")
plt.xlabel("change one gamma")
plt.ylabel("e_mine")
plt.legend(["user0", "user1", "user2", "user3"])
plt.savefig("graphs/change one gamma e_mine.png")