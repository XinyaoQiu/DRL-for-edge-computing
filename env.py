import numpy as np

LAMBDA_E = 0.6
LAMBDA_PHI = 0.4

MU_1 = 0.6
MU_2 = 0.4

K_CHANNEL = 4

MIN_SIZE = 0.2
MAX_SIZE = 50

MIN_CYCLE = 0.05
MAX_CYCLE = 2

MIN_DDL = 0.4
MAX_DDL = 2

MIN_RES = 0.4
MAX_RES = 2.3

MIN_COM = 0.1
MAX_COM = 1

MAX_POWER = 24

MAX_GAIN = 10
MIN_GAIN = 5
 

V_L = 0.125
V_E = 0.13

THETA_L = 1/1600
THETA_E = 1/1700

K_ENERGY_LOCAL = 0.8 * 10**(-3)   #k = 0.8 * 10 ^(-27) * M * G^2#
K_ENERGY_MEC = 0.7 * 10**(-3)

NOISE_VARIANCE = 100

OMEGA = 0.9*10**(-2)  #w = 0.9*10*(-11) * G#

CAPABILITY_E = 5 

MIN_EPSILON = 0.56
MAX_EPSILON = 0.93

KSI = 0.5
LAMBDA = 0.5
ALPHA = 0.5
BETA = 10

S_POWER = 20
S_GAIN = 8
S_SIZE = 8
S_CYCLE = 1
S_RESOLU = 0.6

S_RES = 1.5
S_COM = 0.6


class MecBCEnv(object):
    def __init__(self, n_agents, S_DDL=1, S_EPSILON=0.86, W_BANDWIDTH=20, \
        S_one_power=20, S_one_gamma=0.6, mode="normal"):
        

        self.state_size = 10 
        self.action_size = 6 
        self.n_agents = n_agents

        self.S_DDL = S_DDL
        self.S_EPSILON = S_EPSILON

        self.W_BANDWIDTH = W_BANDWIDTH
        self.S_one_power = S_one_power
        self.S_one_gamma = S_one_gamma

        # state
        self.S_channel = np.zeros(self.n_agents)
        self.S_power = np.zeros(self.n_agents)
        self.S_gain = np.zeros(self.n_agents)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)  
        self.S_resolu = np.zeros(self.n_agents)
        self.S_ddl = np.zeros(self.n_agents)
        self.S_res = np.zeros(self.n_agents)
        self.S_com = np.zeros(self.n_agents)
        self.S_epsilon = np.zeros(self.n_agents)
        self.mode = mode

        self.action_lower_bound = [0, 0, 0.01, MIN_RES, MIN_COM, 1]
        self.action_higher_bound = [1, K_CHANNEL, 0.99, MAX_RES, MAX_COM, MAX_POWER]

          

        self.epoch = 0

    # 重置
    def reset(self):
        self.epoch = 0
        #随机state
        for n in range(self.n_agents):
            self.S_channel[n] = 1
            self.S_power[n] = np.random.normal(S_POWER, 1)
            self.S_gain[n] = np.random.normal(S_GAIN, 1)
            self.S_size[n] =  np.random.normal(S_SIZE, 1)
            self.S_cycle[n] = np.random.normal(S_CYCLE, 0.1)
            self.S_resolu[n] = np.random.normal(S_RESOLU, 0.1)
            self.S_ddl[n] = np.random.normal(self.S_DDL, 0.1)
            self.S_res[n] = np.random.normal(S_RES, 0.1)
            self.S_com[n] = np.random.normal(S_COM, 0.1)
            self.S_epsilon[n] = np.random.normal(self.S_EPSILON, 0.1)

        self.S_power[0] = np.random.normal(self.S_one_power, 1)
        self.S_com[0] = np.random.normal(self.S_one_gamma, 0.1)

        State_ = []
        State_ = [[self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n], \
            self.S_resolu[n], self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n]] for n in range(self.n_agents)]

        State_ = np.array(State_)

   

        return State_


    def step(self, action):
        

        # action
        A_decision = np.zeros(self.n_agents)
        A_channel = np.zeros(self.n_agents)
        A_resolu = np.zeros(self.n_agents)
        A_res = np.zeros(self.n_agents)
        A_com = np.zeros(self.n_agents)
        A_power = np.zeros(self.n_agents)
        if self.mode == "normal":
            for n in range(self.n_agents):
                A_decision[n] = action[n][0] 
                A_channel[n] = action[n][1]
                A_resolu[n] = action[n][2]
                A_res[n] = action[n][3]
                A_com[n] = action[n][4]
                A_power[n] = action[n][5]
        elif self.mode == "NAC":
            for n in range(self.n_agents):
                A_decision[n] = action[n][0] 
                A_channel[n] = action[n][1]
                A_resolu[n] = 0.2
                A_res[n] = action[n][3]
                A_com[n] = action[n][4]
                A_power[n] = action[n][5]
        elif self.mode == "ALLES":
            for n in range(self.n_agents):
                A_decision[n] = 1
                A_channel[n] = action[n][1]
                A_resolu[n] = action[n][2]
                A_res[n] = action[n][3]
                A_com[n] = action[n][4]
                A_power[n] = action[n][5]
        else:
            print("Wrong!")

        
        S_channel = self.S_channel
        S_power = self.S_power
        S_gain = self.S_gain
        S_size = self.S_size
        S_cycle = self.S_cycle
        S_resolu = self.S_resolu
        S_ddl = self.S_ddl
        S_res = self.S_res
        S_com = self.S_com
        S_epsilon = self.S_epsilon
        
        # 根据S_task, S_channel调整A_decision
        for n in range(self.n_agents):
            for k in range(K_CHANNEL):
                if S_channel[n] == k and A_channel[n] == k:
                    A_decision[n] = 0

        # 求reward
        x_n = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            if S_channel[n] != 0:
                x_n[n] = 1
            else:
                x_n[n] = 0

        total_power = 0
        for n in range(self.n_agents):
            total_power += x_n[n] * S_power[n] * S_gain[n]
        

        Phi_local = V_L * np.log(1 + S_resolu / THETA_L) 

        Phi_off = V_E * np.log(1 + S_resolu / THETA_E) 

        Phi_n = (1 - x_n) * Phi_local + x_n * Phi_off 


        Phi_penalty = np.maximum((S_epsilon - Phi_n) / S_epsilon, 0)
        

        total_com = np.sum(S_com)
        
        DataRate  = self.W_BANDWIDTH * np.log(1 + S_power * S_gain / (NOISE_VARIANCE + \
                        total_power - x_n * S_power * S_gain)) / np.log(2) 


        Time_proc = S_resolu * S_cycle / CAPABILITY_E 

        Time_local = S_resolu * S_cycle / S_res 

        Time_off = S_resolu * S_size / DataRate 

        Time_n = (1 - x_n) * Time_local + x_n * (Time_off + Time_proc) 

        total_com = np.sum(S_com)

        T_mean = np.mean(Time_n)

        R_mine = KSI * S_com / total_com * np.exp(-LAMBDA * T_mean / S_ddl) 

        Time_penalty = np.maximum((Time_n - S_ddl) / Time_n, 0)

        Energy_local = K_ENERGY_LOCAL * S_size * S_resolu * (S_res**2) + OMEGA * S_com 

        Energy_off = S_power * Time_off * 10**(-6) 

        Energy_mine = OMEGA * S_com 

        Energy_n = (1 - x_n) * Energy_local + x_n * Energy_off 

        Reward_vt = LAMBDA_E * ((Energy_local - Energy_n) / Energy_local) - LAMBDA_PHI * ((Phi_local - Phi_n) / Phi_local) 
        
        Utility_mine = R_mine - Energy_mine 

        Reward = MU_1 * Reward_vt + MU_2 * Utility_mine  - BETA * (Phi_penalty + Time_penalty)

        # print(np.sum(Reward), np.sum(Reward_mine), np.sum(Reward_vt), np.sum(Phi_penalty), np.sum(Time_penalty))

        # 根据action改state
        for n in range(self.n_agents):
            if int(A_decision[n]):
                self.S_channel[n] = A_channel[n]
        self.S_resolu = A_resolu
        self.S_res = A_res
        self.S_com = A_com
        self.S_power = A_power

        State_ = []
        State_ = [[self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n], \
            self.S_resolu[n], self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n]] for n in range(self.n_agents)]

        State_ = np.array(State_)

        self.epoch += 1
        done = False
        if self.epoch > 100:
            self.reset()
            done = True

        

        return State_, Reward, done, True, Phi_n, Energy_n, R_mine, Energy_mine



