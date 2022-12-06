
from MAPPO import MAPPO
from Model import NUMBER

import matplotlib.pyplot as plt

from env import MecBCEnv

import xlrd
from xlutils.copy import copy as xl_copy

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0


def create_ppo(env, critic_lr=0.001, actor_lr=0.001, noise=0, tau=300):
    ppo = MAPPO(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr, noise=noise, tau=tau) 
    while ppo.n_episodes < MAX_EPISODES:
        ppo.interact()
        if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
            ppo.train()
    return ppo

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

def plot_ppo(ppo, parameter, parameterlist, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ppo)):
            plt.plot(ppo[i].episodes, ppo[i].mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
    elif (variable == "phi"):
        for i in range(len(ppo)):
            plt.plot(ppo[i].episodes, ppo[i].mean_phi) 
            plt.xlabel("Episode")
            plt.ylabel("Phi")
    plt.legend(["%s=%s"%(parameter, parameterlist[i]) for i in range(len(parameterlist))])
    plt.savefig("./output/ppo_change_%s.png"%parameter)

def run(times, variable):
    # All_ddl = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # All_ddl = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    All_epsilon = [0.77, 0.80]
    # All_bandwidth = [20, 40, 60, 80, 100, 120]
    # All_agents = [3]

    # noise = [0, 0]
    # tau = [300, 300]

    rworkbook = xlrd.open_workbook('excel/Excel_ppo.xls', formatting_info=True)
    wworkbook = xl_copy(rworkbook)
    
    # # change ddl
    # env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i]) for i in range(len(All_ddl))]
    # ppo_ddl_list = [create_ppo(env_ddl_list[i], noise=noise[i], tau=tau[i]) for i in range(len(env_ddl_list))]
    # wworkbook = writeExcel(ppo_ddl_list, wworkbook, "Change_ddl_%s"%times, All_ddl, variable)
    # plot_ppo(ppo_ddl_list, "ddl_%s"%times, All_ddl, variable)

    # change epsilon
    env_epsilon_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=All_epsilon[i]) for i in range(len(All_epsilon))]
    ppo_epsilon_list = [create_ppo(env_epsilon_list[i]) for i in range(len(env_epsilon_list))]
    wworkbook = writeExcel(ppo_epsilon_list, wworkbook, "Change_epsilon_%s"%times, All_epsilon, variable)
    plot_ppo(ppo_epsilon_list, "epsilon_%s"%times, All_epsilon, variable)

    # # change bandwidth
    # env_bandwidth_list = [MecBCEnv(n_agents=NUMBER, W_BANDWIDTH=All_bandwidth[i]) for i in range(len(All_bandwidth))]
    # ppo_bandwidth_list = [create_ppo(env_bandwidth_list[i]) for i in range(len(env_bandwidth_list))]
    # wworkbook = writeExcel(ppo_bandwidth_list, wworkbook, "Change_bandwidth_%s"%times, All_bandwidth, variable)
    # plot_ppo(ppo_bandwidth_list, "bandwidth_%s"%times, All_bandwidth, variable)

    # # change agents
    # env_agents_list = [MecBCEnv(n_agents=All_agents[i]) for i in range(len(All_agents))]
    # ppo_agents_list = [create_ppo(env_agents_list[i], noise=noise[i], tau=tau[i]) for i in range(len(env_agents_list))]
    # wworkbook = writeExcel(ppo_agents_list, wworkbook, "Change_agents_%s"%times, All_agents, variable)
    # plot_ppo(ppo_agents_list, "agents_%s"%times, All_agents, variable)

    wworkbook.save('excel/Excel_ppo.xls')

if __name__ == "__main__":
    run(2, "reward")
