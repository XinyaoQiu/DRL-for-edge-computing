from MADDPG import MADDPG
from Model import NUMBER

import matplotlib.pyplot as plt
from env import MecBCEnv

import xlrd
from xlutils.copy import copy as xl_copy


MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 100  
EVAL_EPISODES = 10
EVAL_INTERVAL = 10

# max steps in each episode, prevent from running too long
MAX_STEPS = 10000 # None

MEMORY_CAPACITY = 10000
BATCH_SIZE = 100
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

TARGET_UPDATE_STEPS = 5
TARGET_TAU = 0.01

REWARD_DISCOUNTED_GAMMA = 0.99

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

DONE_PENALTY = None

RANDOM_SEED = 2022


def create_ddpg(env, critic_lr=0.001, actor_lr=0.001):
    ddpg = MADDPG(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr) 
    while ddpg.n_episodes < MAX_EPISODES:
        ddpg.interact()
        # if ddpg.n_episodes >= EPISODES_BEFORE_TRAIN:
        #     ddpg.train()
    return ddpg
    

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
            for n in range(NUMBER):
                sheet.write(0, NUMBER*i+n+1, "Phi(%s=%.2f)(user%s)" %(sheetname, parameterlist[i], n))
                for j in range(len(agent[i].episodes)):
                    # row, column, value
                    sheet.write(j+1, NUMBER*i+n+1, agent[i].mean_phi[n][j])
        elif (variable == "energy"):
            for n in range(NUMBER):
                sheet.write(0, NUMBER*i+n+1, "Energy(%s=%.2f)(user%s)" %(sheetname, parameterlist[i], n))
                for j in range(len(agent[i].episodes)):
                    # row, column, value
                    sheet.write(j+1, NUMBER*i+n+1, agent[i].mean_energy[n][j])
        elif (variable == "agent_reward"):
            for n in range(NUMBER):
                sheet.write(0, NUMBER*i+n+1, "Reward(%s=%.2f)(user%s)" %(sheetname, parameterlist[i], n))
                for j in range(len(agent[i].episodes)):
                    # row, column, value
                    sheet.write(j+1, NUMBER*i+n+1, agent[i].agent_mean_rewards[n][j])
    return workbook

def plot_ddpg(ddpg, parameter, parameterlist, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
    elif (variable == "phi"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_phi) 
            plt.xlabel("Episode")
            plt.ylabel("Phi")
    elif (variable == "energy"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_energy) 
            plt.xlabel("Episode")
            plt.ylabel("Energy")
    elif (variable == "agent_reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].agent_mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")        
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["%s=%s"%(parameter, parameterlist[i]) for i in range(len(parameterlist))])
    plt.savefig("./output/ddpg_change_%s.png"%parameter)

def run(times):
    All_ddl = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    All_epsilon = [0.77, 0.80, 0.83, 0.86, 0.90, 0.93]
    All_bandwidth = [20, 40, 60, 80, 100, 120]
    All_agents = [1, 2, 3, 4, 5, 6]

    rworkbook = xlrd.open_workbook('excel/Excel_ddpg.xls', formatting_info=True)
    wworkbook = xl_copy(rworkbook)

    # change ddl
    # env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i]) for i in range(len(All_ddl))]
    # ddpg_ddl_list = [create_ddpg(env_ddl_list[i]) for i in range(len(env_ddl_list))]
    # wworkbook = writeExcel(ddpg_ddl_list, wworkbook, "change_one_power2", All_ddl, "agent_reward")
    # plot_ddpg(ddpg_ddl_list, "ddl_%s"%times, All_ddl)

    # # change epsilon
    # env_epsilon_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=All_epsilon[i]) for i in range(len(All_epsilon))]
    # ddpg_epsilon_list = [create_ddpg(env_epsilon_list[i]) for i in range(len(env_epsilon_list))]
    # wworkbook = writeExcel(ddpg_epsilon_list, wworkbook, "Change_epsilon_%s"%times, All_epsilon)
    # plot_ddpg(ddpg_epsilon_list, "epsilon_%s"%times, All_epsilon)

    # # change bandwidth
    # env_bandwidth_list = [MecBCEnv(n_agents=NUMBER, W_BANDWIDTH=All_bandwidth[i]) for i in range(len(All_bandwidth))]
    # ddpg_bandwidth_list = [create_ddpg(env_bandwidth_list[i]) for i in range(len(env_bandwidth_list))]
    # wworkbook = writeExcel(ddpg_bandwidth_list, wworkbook, "Change_bandwidth_%s"%times, All_bandwidth, variable)
    # plot_ddpg(ddpg_bandwidth_list, "bandwidth_%s"%times, All_bandwidth, variable)

    # # change agents
    # env_agents_list = [MecBCEnv(n_agents=All_agents[i]) for i in range(len(All_agents))]
    # ddpg_agents_list = [create_ddpg(env_agents_list[i]) for i in range(len(env_agents_list))]
    # wworkbook = writeExcel(ddpg_agents_list, wworkbook, "Change_agents_%s"%times, All_agents, variable)
    # plot_ddpg(ddpg_agents_list, "agents_%s"%times, All_agents, variable)

    # change one power
    # All_one_power = [40, 60, 80, 100, 120, 140]
    All_one_gamma = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    # env_ddl_list1 = [MecBCEnv(n_agents=NUMBER, S_one_power=All_one_power[i]) for i in range(len(All_one_power))]
    # ddpg_ddl_list1 = [create_ddpg(env_ddl_list1[i]) for i in range(len(env_ddl_list1))]
    # wworkbook = writeExcel(ddpg_ddl_list1, wworkbook, "change_one_power_phi_%s"%times, All_one_power, "phi")
    # wworkbook = writeExcel(ddpg_ddl_list1, wworkbook, "change_one_power_energy_%s"%times, All_one_power, "energy") 
    # env_ddl_list2 = [MecBCEnv(n_agents=NUMBER, S_one_gamma=All_one_gamma[i]) for i in range(len(All_one_gamma))]
    # ddpg_ddl_list2 = [create_ddpg(env_ddl_list2[i]) for i in range(len(env_ddl_list2))]
    # wworkbook = writeExcel(ddpg_ddl_list2, wworkbook, "change_one_gamma_r_mine_%s"%times, All_one_gamma, "phi")
    # wworkbook = writeExcel(ddpg_ddl_list2, wworkbook, "change_one_gamma_e_mine_%s"%times, All_one_gamma, "energy") 

    # change ddl
    env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i], mode="ALLES") for i in range(len(All_ddl))]
    ddpg_ddl_list = [create_ddpg(env_ddl_list[i]) for i in range(len(env_ddl_list))]
    wworkbook = writeExcel(ddpg_ddl_list, wworkbook, "ALLES_ddl", All_ddl)
    plot_ddpg(ddpg_ddl_list, "ddl_%s"%times, All_ddl)

    # # change epsilon
    env_epsilon_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=All_epsilon[i], mode="ALLES") for i in range(len(All_epsilon))]
    ddpg_epsilon_list = [create_ddpg(env_epsilon_list[i]) for i in range(len(env_epsilon_list))]
    wworkbook = writeExcel(ddpg_epsilon_list, wworkbook, "ALLES_epsilon", All_epsilon)
    # plot_ddpg(ddpg_epsilon_list, "epsilon_%s"%times, All_epsilon)

    # # change bandwidth
    env_bandwidth_list = [MecBCEnv(n_agents=NUMBER, W_BANDWIDTH=All_bandwidth[i], mode="ALLES") for i in range(len(All_bandwidth))]
    ddpg_bandwidth_list = [create_ddpg(env_bandwidth_list[i]) for i in range(len(env_bandwidth_list))]
    wworkbook = writeExcel(ddpg_bandwidth_list, wworkbook, "ALLES_bandwidth", All_bandwidth)
    # plot_ddpg(ddpg_bandwidth_list, "bandwidth_%s"%times, All_bandwidth, variable)

    # # change agents
    env_agents_list = [MecBCEnv(n_agents=All_agents[i], mode="ALLES") for i in range(len(All_agents))]
    ddpg_agents_list = [create_ddpg(env_agents_list[i]) for i in range(len(env_agents_list))]
    wworkbook = writeExcel(ddpg_agents_list, wworkbook, "ALLES_agents", All_agents)
    # plot_ddpg(ddpg_agents_list, "agents_%s"%times, All_agents)

    wworkbook.save('excel/Excel_ddpg.xls')

if __name__ == "__main__":
    run(5)
