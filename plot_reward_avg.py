import matplotlib.pyplot as plt
import xlrd

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0

#plot reward_vs_ddl
rworkbook = xlrd.open_workbook("excel/final.xls")
sheet_a2c = rworkbook.sheet_by_name("a2c_ddl")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_ddl")
sheet_ppo = rworkbook.sheet_by_name("ppo_ddl")
sheet_NAC = rworkbook.sheet_by_name("NAC_ddl")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_ddl")

ddls = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
reward_a2c = [sheet_a2c.cell(201, i+1).value for i in range(6)]
reward_ddpg = [sheet_ddpg.cell(201, i+1).value for i in range(6)]
reward_ppo = [sheet_ppo.cell(201, i+1).value for i in range(6)]
reward_NAC = [sheet_NAC.cell(201, i+1).value for i in range(6)]
reward_ALLES = [sheet_ALLES.cell(201, i+1).value for i in range(6)]

plt.figure()
plt.plot(ddls, reward_a2c)
plt.plot(ddls, reward_ddpg)
plt.plot(ddls, reward_ppo) 
plt.plot(ddls, reward_NAC) 
plt.plot(ddls, reward_ALLES) 
plt.xlabel("DDL")
plt.ylabel("Reward")
plt.legend(["MAA2C", "MADDPG", "MAPPO", "NAC", "ALLES"])
plt.savefig("graphs/reward_vs_parameters/reward_vs_ddl.png")
plt.close()

#plot reward_vs_agents
sheet_a2c = rworkbook.sheet_by_name("a2c_agents")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_agents")
sheet_ppo = rworkbook.sheet_by_name("ppo_agents")
sheet_NAC = rworkbook.sheet_by_name("NAC_agents")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_agents")

agents = [1, 2, 3, 4, 5, 6]
reward_a2c = [sheet_a2c.cell(201, i+1).value for i in range(6)]
reward_ddpg = [sheet_ddpg.cell(201, i+1).value for i in range(6)]
reward_ppo = [sheet_ppo.cell(201, i+1).value for i in range(6)]
reward_NAC = [sheet_NAC.cell(201, i+1).value for i in range(6)]
reward_ALLES = [sheet_ALLES.cell(201, i+1).value for i in range(6)]

plt.figure()
plt.plot(agents, reward_a2c)
plt.plot(agents, reward_ddpg)
plt.plot(agents, reward_ppo)
plt.plot(agents, reward_NAC)
plt.plot(agents, reward_ALLES)
plt.xlabel("Agents Number")
plt.ylabel("Reward")
plt.legend(["MAA2C", "MADDPG", "MAPPO", "NAC", "ALLES"])
plt.savefig("graphs/reward_vs_parameters/reward_vs_agents.png")
plt.close()

#plot reward_vs_bandwidth
sheet_a2c = rworkbook.sheet_by_name("a2c_bandwidth")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_bandwidth")
sheet_ppo = rworkbook.sheet_by_name("ppo_bandwidth")
sheet_NAC = rworkbook.sheet_by_name("NAC_bandwidth")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_bandwidth")

bandwidths = [20, 40, 60, 80, 100, 120]
reward_a2c = [sheet_a2c.cell(201, i+1).value for i in range(6)]
reward_ddpg = [sheet_ddpg.cell(201, i+1).value for i in range(6)]
reward_ppo = [sheet_ppo.cell(201, i+1).value for i in range(6)]
reward_NAC = [sheet_NAC.cell(201, i+1).value for i in range(6)]
reward_ALLES = [sheet_ALLES.cell(201, i+1).value for i in range(6)]

plt.figure()
plt.plot(bandwidths, reward_a2c)
plt.plot(bandwidths, reward_ddpg)
plt.plot(bandwidths, reward_ppo)
plt.plot(bandwidths, reward_NAC)
plt.plot(bandwidths, reward_ALLES)
plt.xlabel("Bandwidth")
plt.ylabel("Reward")
plt.legend(["MAA2C", "MADDPG", "MAPPO", "NAC", "ALLES"])
plt.savefig("graphs/reward_vs_parameters/reward_vs_bandwidth.png")
plt.close()

#plot reward_vs_epsilon
sheet_a2c = rworkbook.sheet_by_name("a2c_epsilon")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_epsilon")
sheet_ppo = rworkbook.sheet_by_name("ppo_epsilon")
sheet_NAC = rworkbook.sheet_by_name("NAC_epsilon")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_epsilon")

epsilons = [0.77, 0.80, 0.83, 0.86, 0.90, 0.93]
reward_a2c = [sheet_a2c.cell(201, i+1).value for i in range(6)]
reward_ddpg = [sheet_ddpg.cell(201, i+1).value for i in range(6)]
reward_ppo = [sheet_ppo.cell(201, i+1).value for i in range(6)]
reward_NAC = [sheet_NAC.cell(201, i+1).value for i in range(6)]
reward_ALLES = [sheet_ALLES.cell(201, i+1).value for i in range(6)]

plt.figure()
plt.plot(epsilons, reward_a2c)
plt.plot(epsilons, reward_ddpg)
plt.plot(epsilons, reward_ppo)
plt.plot(epsilons, reward_NAC)
plt.plot(epsilons, reward_ALLES)
plt.xlabel("Epsilon")
plt.ylabel("Reward")
plt.legend(["MAA2C", "MADDPG", "MAPPO", "NAC", "ALLES"])
plt.savefig("graphs/reward_vs_parameters/reward_vs_epsilon.png")
plt.close()