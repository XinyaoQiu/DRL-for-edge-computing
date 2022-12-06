from MAA2C import MAA2C
from MADDPG import MADDPG
from MAPPO import MAPPO
from Model import NUMBER
from env import MecBCEnv

import matplotlib.pyplot as plt

import xlrd
from xlutils.copy import copy as xl_copy

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0


def create_a2c(env, critic_lr=0.001, actor_lr=0.001):
    a2c = MAA2C(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr) 
    while a2c.n_episodes < MAX_EPISODES:
        a2c.interact()
        if a2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            a2c.train()
    return a2c

def create_ddpg(env, critic_lr=0.001, actor_lr=0.001):
    ddpg = MADDPG(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr) 
    while ddpg.n_episodes < MAX_EPISODES:
        ddpg.interact()
        if ddpg.n_episodes >= EPISODES_BEFORE_TRAIN:
            ddpg.train()
    return ddpg


def create_ppo(env, critic_lr=0.001, actor_lr=0.001):
    ppo = MAPPO(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
                  critic_lr=critic_lr, actor_lr=actor_lr) 
    while ppo.n_episodes < MAX_EPISODES:
        ppo.interact()
        if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
            ppo.train()
    return ppo

def writeExcel(agent, workbook, sheetname, parameterlist):
    #REQUIRE: agent list
    sheet = workbook.add_sheet(sheetname)
    sheet.write(0, 0, "Episodes")
    for j in range(len(agent[0].episodes)):
        sheet.write(j+1, 0, agent[0].episodes[j])
    for i in range(len(parameterlist)):
        sheet.write(0, i+1, "Rewards(%s=%.2f)" %(sheetname, parameterlist[i])) 
        
        for j in range(len(agent[i].episodes)):
             # row, column, value
            sheet.write(j+1, i+1, agent[i].mean_rewards[j])
    
    return workbook

def plot_from_excel(sheet):
    plt.figure()
    episodes = []
    rewards_ddpg = []
    rewards_a2c = []
    rewards_ppo = []
    for i in range(1, sheet.nrows):
        episodes.append(sheet.cell(i, 0).value)
        rewards_ddpg.append(sheet.cell(i, 1).value)
        rewards_a2c.append(sheet.cell(i, 2).value)
        rewards_ppo.append(sheet.cell(i, 3).value)

    plt.plot(episodes, rewards_ddpg)
    plt.plot(episodes, rewards_a2c)
    plt.plot(episodes, rewards_ppo)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DDPG", "A2C", "PPO"])

    plt.savefig("./output/comparison.png")

def run():
    All_ddl = [1]
    env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i]) for i in range(len(All_ddl))]
    
    # ddpg_ddl_list = [create_ddpg(env_ddl_list[i]) for i in range(len(env_ddl_list))]
    a2c_ddl_list = [create_a2c(env_ddl_list[i]) for i in range(len(env_ddl_list))]
    ppo_ddl_list = [create_ppo(env_ddl_list[i]) for i in range(len(env_ddl_list))]

    rworkbook = xlrd.open_workbook('DDPG_A2C_PPO.xls', formatting_info=True)
    wworkbook = xl_copy(rworkbook)
    # workbook = writeExcel(ddpg_ddl_list, wworkbook, "DDPG", All_ddl)
    workbook = writeExcel(a2c_ddl_list, wworkbook, "A2C", All_ddl)

    workbook = writeExcel(ppo_ddl_list, wworkbook, "PPO", All_ddl)
    workbook.save('DDPG_A2C_PPO.xls')

def plot():
    rworkbook = xlrd.open_workbook('DDPG_A2C_PPO.xls', formatting_info=True)
    sheet = rworkbook.sheet_by_name("Plot")
    plot_from_excel(sheet)

if __name__ == "__main__":
    # run()
    plot()
    
    
