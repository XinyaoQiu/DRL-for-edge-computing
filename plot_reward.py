import matplotlib.pyplot as plt
import xlrd

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0

#plot reward_vs_ddl

def plot_reward(parameter, sheet, model, paraname):
    episodes = [sheet.cell(i+1, 0).value for i in range(200)]
    reward0 = [sheet.cell(i+1, 1).value for i in range(200)]
    reward1 = [sheet.cell(i+1, 2).value for i in range(200)]
    reward2 = [sheet.cell(i+1, 3).value for i in range(200)]
    reward3 = [sheet.cell(i+1, 4).value for i in range(200)]
    reward4 = [sheet.cell(i+1, 5).value for i in range(200)]
    reward5 = [sheet.cell(i+1, 6).value for i in range(200)]
    plt.figure()
    plt.plot(episodes, reward0)
    plt.plot(episodes, reward1)
    plt.plot(episodes, reward2)
    plt.plot(episodes, reward3)
    plt.plot(episodes, reward4)
    plt.plot(episodes, reward5)
    plt.xlabel("episodes")
    plt.ylabel(model)
    plt.legend(["%s=%s"%(paraname, i) for i in parameter])
    plt.savefig("graphs/change %s/%s_change_%s.png"%(paraname, model, paraname))
    plt.close()

rworkbook = xlrd.open_workbook("excel/final.xls")

ddls = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
sheet_a2c = rworkbook.sheet_by_name("a2c_ddl")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_ddl")
sheet_ppo = rworkbook.sheet_by_name("ppo_ddl")
sheet_NAC = rworkbook.sheet_by_name("NAC_ddl")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_ddl")

plot_reward(ddls, sheet_a2c, "a2c", "ddl")
plot_reward(ddls, sheet_ddpg, "ddpg", "ddl")
plot_reward(ddls, sheet_ppo, "ppo", "ddl")
plot_reward(ddls, sheet_NAC, "NAC", "ddl")
plot_reward(ddls, sheet_ALLES, "ALLES", "ddl")

agents = [1, 2, 3, 4, 5, 6]
sheet_a2c = rworkbook.sheet_by_name("a2c_agents")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_agents")
sheet_ppo = rworkbook.sheet_by_name("ppo_agents")
sheet_NAC = rworkbook.sheet_by_name("NAC_agents")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_agents")

plot_reward(agents, sheet_a2c, "a2c", "agents")
plot_reward(agents, sheet_ddpg, "ddpg", "agents")
plot_reward(agents, sheet_ppo, "ppo", "agents")
plot_reward(agents, sheet_NAC, "NAC", "agents")
plot_reward(agents, sheet_ALLES, "ALLES", "agents")

All_epsilon = [0.77, 0.80, 0.83, 0.86, 0.90, 0.93]
sheet_a2c = rworkbook.sheet_by_name("a2c_epsilon")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_epsilon")
sheet_ppo = rworkbook.sheet_by_name("ppo_epsilon")
sheet_NAC = rworkbook.sheet_by_name("NAC_epsilon")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_epsilon")

plot_reward(All_epsilon, sheet_a2c, "a2c", "epsilon")
plot_reward(All_epsilon, sheet_ddpg, "ddpg", "epsilon")
plot_reward(All_epsilon, sheet_ppo, "ppo", "epsilon")
plot_reward(All_epsilon, sheet_NAC, "NAC", "epsilon")
plot_reward(All_epsilon, sheet_ALLES, "ALLES", "epsilon")

All_bandwidth = [20, 40, 60, 80, 100, 120]
sheet_a2c = rworkbook.sheet_by_name("a2c_bandwidth")
sheet_ddpg = rworkbook.sheet_by_name("ddpg_bandwidth")
sheet_ppo = rworkbook.sheet_by_name("ppo_bandwidth")
sheet_NAC = rworkbook.sheet_by_name("NAC_bandwidth")
sheet_ALLES = rworkbook.sheet_by_name("ALLES_bandwidth")

plot_reward(All_bandwidth, sheet_a2c, "a2c", "bandwidth")
plot_reward(All_bandwidth, sheet_ddpg, "ddpg", "bandwidth")
plot_reward(All_bandwidth, sheet_ppo, "ppo", "bandwidth")
plot_reward(All_bandwidth, sheet_NAC, "NAC", "bandwidth")
plot_reward(All_bandwidth, sheet_ALLES, "ALLES", "bandwidth")

