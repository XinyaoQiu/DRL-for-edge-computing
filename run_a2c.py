from MAA2C import MAA2C
from Model import NUMBER
from env import MecBCEnv

import matplotlib.pyplot as plt

import xlrd
from xlutils.copy import copy as xl_copy

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0


def create_a2c(env, critic_lr=0.001, actor_lr=0.001, noise=0.04, tau=1400, bound=600):
    a2c = MAA2C(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr, noise=noise, tau=tau, bound=bound) 
    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            a2c.train()
    return a2c

def writeExcel(agent, workbook, sheetname, parameterlist, variable="reward"):
    #REQUIRE: agent list
    sheet = workbook.add_sheet(sheetname)
    sheet.write(0, 0, "Episodes")
    for j in range(len(agent[0].episodes)):
        sheet.write(j+1, 0, agent[0].episodes[j])
    for i in range(len(parameterlist)):
        if (variable == "reward"):
            sheet.write(0, i+1, "Rewards(%s=%.2f)" %(sheetname, parameterlist[i]))
            for j in range(len(agent[i].episodes)):
                # row, column, value
                sheet.write(j+1, i+1, agent[i].mean_rewards[j])
        elif (variable == "phi"):
            sheet.write(0, i+1, "Phi(%s=%.2f)" %(sheetname, parameterlist[i]))
            for j in range(len(agent[i].episodes)):
                # row, column, value
                sheet.write(j+1, i+1, agent[i].mean_phi[j])
    return workbook

def plot_a2c(a2c, parameter, parameterlist, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(a2c)):
            plt.plot(a2c[i].episodes, a2c[i].mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
    elif (variable == "phi"):
        for i in range(len(a2c)):
            plt.plot(a2c[i].episodes, a2c[i].mean_phi) 
            plt.xlabel("Episode")
            plt.ylabel("Phi")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["%s=%s"%(parameter, parameterlist[i]) for i in range(len(parameterlist))])
    plt.savefig("./output/a2c_change_%s.png"%parameter)

def run(times, variable):
    All_ddl = [0.9, 1.0, 1.1]
    # All_epsilon = [0.83, 0.86, 0.9, 0.93]
    # All_bandwidth = [100, 200, 300, 400, 500, 600, 700]
    # All_agents = [3, 4, 5, 6]

    noise = [0.05, 0.05, 0.05]
    tau = [500, 1000, 2000]
    bound = [2200, 2200, 2200]

    

    rworkbook = xlrd.open_workbook('excel/Excel_a2c.xls', formatting_info=True)
    wworkbook = xl_copy(rworkbook)
    
    # change ddl
    env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i]) for i in range(len(All_ddl))]
    a2c_ddl_list = [create_a2c(env_ddl_list[i], noise=noise[i], tau=tau[i], bound=bound[i]) for i in range(len(env_ddl_list))]
    wworkbook = writeExcel(a2c_ddl_list, wworkbook, "Change_ddl_%s"%times, All_ddl, variable)
    plot_a2c(a2c_ddl_list, "ddl_%s"%times, All_ddl, variable)

    # # change epsilon
    # env_epsilon_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=All_epsilon[i]) for i in range(len(All_epsilon))]
    # a2c_epsilon_list = [create_a2c(env_epsilon_list[i]) for i in range(len(env_epsilon_list))]
    # wworkbook = writeExcel(a2c_epsilon_list, wworkbook, "Change_epsilon_%s"%times, All_epsilon, variable)
    # plot_a2c(a2c_epsilon_list, "epsilon_%s"%times, All_epsilon, variable)

    # # change bandwidth
    # env_bandwidth_list = [MecBCEnv(n_agents=NUMBER, W_BANDWIDTH=All_bandwidth[i]) for i in range(len(All_bandwidth))]
    # a2c_bandwidth_list = [create_a2c(env_bandwidth_list[i]) for i in range(len(env_bandwidth_list))]
    # wworkbook = writeExcel(a2c_bandwidth_list, wworkbook, "Change_bandwidth_%s"%times, All_bandwidth, variable)
    # plot_a2c(a2c_bandwidth_list, "bandwidth_%s"%times, All_bandwidth, variable)

    # # change agents
    # env_agents_list = [MecBCEnv(n_agents=All_agents[i]) for i in range(len(All_agents))]
    # a2c_agents_list = [create_a2c(env_agents_list[i], noise[i], tau[i]) for i in range(len(env_agents_list))]
    # wworkbook = writeExcel(a2c_agents_list, wworkbook, "Change_agents_%s"%times, All_agents, variable)
    # plot_a2c(a2c_agents_list, "agents_%s"%times, All_agents, variable)

    wworkbook.save('excel/Excel_a2c.xls')

if __name__ == "__main__":
    run(20, "reward")
