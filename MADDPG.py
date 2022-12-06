import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

from utils import to_tensor_var
from Model import ActorNetwork, CriticNetwork
from Memory import ReplayMemory

EVAL_EPISODES = 10


class MADDPG(object):
    """
    An agent learned with Deep Deterministic Policy Gradient using Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - Critic uses gradient temporal-difference learning
    """
    def __init__(self, env, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 memory_capacity=10000, max_steps=10000, target_tau=0.01, target_update_steps=500,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None, training_strategy="centralized",
                 actor_output_act=torch.tanh, actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.01, max_grad_norm=None, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=100, use_cuda=False):

        self.n_agents = n_agents
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound

        self.env_state = env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_output_act = actor_output_act
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.target_tau = target_tau
        self.target_update_steps = target_update_steps

        assert training_strategy in ["cocurrent", "centralized"]
        self.training_strategy = training_strategy

        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_act)] * self.n_agents
        if self.training_strategy == "cocurrent":
            self.critics = [CriticNetwork(self.state_dim, self.action_dim, 1)] * self.n_agents
        elif self.training_strategy == "centralized":
            critic_state_dim = self.n_agents * self.state_dim
            critic_action_dim = self.n_agents * self.action_dim
            self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, 1)] * self.n_agents

        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.critics[i].cuda()
                self.actors_target[i].cuda()
                self.critics_target[i].cuda()

        self.eval_rewards = []
        self.mean_rewards = []
        self.episodes = []
        self.mean_phi = [[] for n in range(self.n_agents)]
        self.eval_phi = [[] for n in range(self.n_agents)]
        self.mean_energy = [[] for n in range(self.n_agents)]
        self.eval_energy = [[] for n in range(self.n_agents)]
        self.mean_R_mine = [[] for n in range(self.n_agents)]
        self.eval_R_mine = [[] for n in range(self.n_agents)]
        self.mean_E_mine = [[] for n in range(self.n_agents)]
        self.eval_E_mine = [[] for n in range(self.n_agents)]
        self.agent_rewards = [[] for n in range(self.n_agents)]
        self.agent_mean_rewards = [[] for n in range(self.n_agents)]

    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        action = self.exploration_action(state)

        next_state, reward, done, _, phi, energy, r_mine, e_mine = self.env.step(action)
        if done:
            if self.done_penalty is not None:
                reward = self.done_penalty
            next_state = np.zeros((self.n_agents, self.state_dim))
            self.env_state = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.env_state = next_state
            self.episode_done = False
        self.n_steps += 1

        # use actor_target to get next_action
        next_state_var = to_tensor_var([next_state], self.use_cuda)
        next_action = np.zeros((self.n_agents, self.action_dim))
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_state_var[:,agent_id,:])
            if self.use_cuda:
                next_action[agent_id] = next_action_var.data.cpu().numpy()[0]
            else:
                next_action[agent_id] = next_action_var.data.numpy()[0]

        self.eval_rewards.append(np.sum(reward))
        for agent_id in range(self.n_agents):
            self.eval_phi[agent_id].append(phi[agent_id])
            self.eval_energy[agent_id].append(energy[agent_id])
            self.eval_R_mine[agent_id].append(r_mine[agent_id])
            self.eval_E_mine[agent_id].append(e_mine[agent_id])
            self.agent_rewards[agent_id].append(reward[agent_id])
        if self.episode_done and ((self.n_episodes+1)%EVAL_EPISODES == 0):
            mean_reward = np.mean(np.array(self.eval_rewards))
            self.mean_rewards.append(mean_reward)
            for agent_id in range(self.n_agents): 
                self.mean_phi[agent_id].append(np.mean(np.array(self.eval_phi[agent_id])))
                self.mean_energy[agent_id].append(np.mean(np.array(self.eval_energy[agent_id])))
                self.mean_R_mine[agent_id].append(np.mean(np.array(self.eval_R_mine[agent_id])))
                self.mean_E_mine[agent_id].append(np.mean(np.array(self.eval_E_mine[agent_id])))
                self.agent_mean_rewards[agent_id].append(np.mean(np.array(self.agent_rewards[agent_id])))
            self.episodes.append(self.n_episodes+1)
            print("Episode:", self.n_episodes+1, "  Average Reward: ", mean_reward)
            self.eval_rewards = []
            self.agent_rewards = [[] for n in range(self.n_agents)]
            self.eval_phi = [[] for n in range(self.n_agents)]
            self.eval_energy = [[] for n in range(self.n_agents)]
            self.eval_R_mine = [[] for n in range(self.n_agents)]
            self.eval_E_mine = [[] for n in range(self.n_agents)]
            
        self.memory.push(state, action, reward, next_state, next_action, done)

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    # train on a sample batch
    def train(self):
        # do not train until exploration is enough
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        next_actions_var = to_tensor_var(batch.next_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)
        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actions_var = actions_var.view(-1, self.n_agents*self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents*self.state_dim)
        whole_next_actions_var = next_actions_var.view(-1, self.n_agents*self.action_dim)

 
        for agent_id in range(self.n_agents): 
            # estimate the target q with actor_target network and critic_target network
            #next_q  (centralized)
            next_q = self.critics_target[agent_id](whole_next_states_var, whole_next_actions_var).detach()

            target_q = self.reward_scale * rewards_var[:,agent_id,:] + self.reward_gamma * next_q * (1. - dones_var)

            # update critic network

            # current Q values (centralized)
            current_q = self.critics[agent_id](whole_states_var, whole_actions_var).detach()

            # rewards is target Q values
            critic_loss = nn.MSELoss()(current_q, target_q)
            critic_loss.requires_grad_(True)
            self.critics_optimizer[agent_id].zero_grad()
            critic_loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critics_optimizer[agent_id].step()

            # update actor network
            
            # the accurate action prediction
            action = self.actors[agent_id](states_var[:,agent_id,:])
            # actor_loss is used to maximize the Q value for the predicted action
            actor_loss = - self.critics[agent_id](whole_states_var, whole_actions_var).detach()
            actor_loss = actor_loss.mean()
            actor_loss.requires_grad_(True)
            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()

            # update actor target network and critic target network
            if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
                self._soft_update_target(self.critics_target[agent_id], self.critics[agent_id])
                self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = np.zeros((self.n_agents, self.action_dim))
        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](state_var[:,agent_id,:])
            if self.use_cuda:
                action[agent_id] = action_var.data.cpu().numpy()[0]
            else:
                action[agent_id] = action_var.data.numpy()[0]
        
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_episodes / self.epsilon_decay)
        # add noise
        noise = np.random.randn(self.n_agents, self.action_dim) * epsilon
        action += noise

        for n in range(self.n_agents):
            for i in range(6):
                if action[n][i] < -1:
                    action[n][i] = -1
                if action[n][i] > 1:
                    action[n][i] = 1
        #get bounded to action_bound
        b = 1
        a = -b
        if self.action_dim > 6:
            print("Wrong!")
        for n in range(self.n_agents):
            action[n][0] = 0 if action[n][0] <= 0 else 1
            action[n][1] = round(self.getactionbound(a, b, action[n][1], 1))
            action[n][2] = self.getactionbound(a, b, action[n][2], 2)
            action[n][3] = self.getactionbound(a, b, action[n][3], 3)
            action[n][4] = self.getactionbound(a, b, action[n][4], 4)
            action[n][5] = self.getactionbound(a, b, action[n][5], 5)
        return action


    # choose an action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = np.zeros((self.n_agents, self.action_dim))
        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](state_var[:,agent_id,:])
            if self.use_cuda:
                action[agent_id] = action_var.data.cpu().numpy()[0]
            else:
                action[agent_id] = action_var.data.numpy()[0]
        
        #get bounded to action_bound
        b = 1
        a = -b
        if self.action_dim > 6:
            print("Wrong!")
        for n in range(self.n_agents):
            action[n][0] = 0 if action[n][0] <= 0 else 1
            action[n][1] = round(self.getactionbound(a, b, action[n][1], 1))
            action[n][2] = self.getactionbound(a, b, action[n][2], 2)
            action[n][3] = self.getactionbound(a, b, action[n][3], 3)
            action[n][4] = self.getactionbound(a, b, action[n][4], 4)
            action[n][5] = self.getactionbound(a, b, action[n][5], 5)

        return action

