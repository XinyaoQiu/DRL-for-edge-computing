
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
from math import exp

import numpy as np
from copy import deepcopy

from Model import ActorNetwork, CriticNetwork
from utils import to_tensor_var
from Memory import ReplayMemory

EVAL_EPISODES = 10

class MAPPO(object):
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """
    def __init__(self, env, state_dim, action_dim, n_agents, action_lower_bound, action_higher_bound,
                 noise=0, tau=300,
                 memory_capacity=10, max_steps=None,
                 roll_out_n_steps=10, target_tau=1.0,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_output_act=nn.functional.softmax, critic_loss="mse",
                 actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.00,
                 max_grad_norm=None, batch_size=10, episodes_before_train=0,
                 use_cuda=False):
        

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.n_agents = n_agents

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale

        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound

        self.memory = ReplayMemory(memory_capacity)

        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.noise = noise
        self.tau = tau

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_act)] * self.n_agents
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
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()
        self.eval_rewards = []
        self.mean_rewards = []
        self.episodes = []
        self.eval_phi = []
        self.mean_phi = []

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset() 
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.choose_action(self.env_state)
            next_state, reward, done, _, phi = self.env.step(action)
            # done = done[0]
            actions.append(action)
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.choose_action(final_state)
            final_r = self.value(final_state, final_action)

        rewards = np.array(rewards)
        for agent_id in range(self.n_agents):
            rewards[:,agent_id] = self._discount_reward(rewards[:,agent_id], final_r[agent_id])
        rewards = rewards.tolist()
        self.n_steps += 1

        self.eval_rewards.append(np.sum(reward))
        self.eval_phi.append(np.sum(phi))
        if self.episode_done and ((self.n_episodes+1)%EVAL_EPISODES == 0):
            mean_reward = np.mean(np.array(self.eval_rewards))
            self.mean_rewards.append(mean_reward)
            self.mean_phi.append(np.mean(np.array(self.eval_phi)))
            self.episodes.append(self.n_episodes+1)
            print("Episode:", self.n_episodes+1, "  Average Reward: ", mean_reward)
            self.eval_rewards = []
            self.eval_phi = []

        self.memory.push(states, actions, rewards)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actions_var = actions_var.view(-1, self.n_agents*self.action_dim)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actors_optimizer[agent_id].zero_grad()
            values = self.critics[agent_id](whole_states_var, whole_actions_var).detach()
            advantages = rewards_var[:,agent_id,:] - values
            # # normalizing advantages seems not working correctly here
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            action_log_probs = self.actors[agent_id](states_var[:,agent_id,:]).detach()
            action_log_probs = th.sum(action_log_probs * actions_var[:,agent_id,:], 1)
            old_action_log_probs = self.actors_target[agent_id](states_var[:,agent_id,:]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:,agent_id,:], 1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.requires_grad_(True)
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()

            # update critic network
            self.critics_optimizer[agent_id].zero_grad()
            target_values = rewards_var[:,agent_id,:]
            # if self.critic_loss == "huber":
            #     critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            # else:
            #     critic_loss = nn.MSELoss()(values, target_values)
            critic_loss = 0.5 * (values - target_values).pow(2).mean()
            critic_loss.requires_grad_(True)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critics_optimizer[agent_id].step()

            # update actor target network and critic target network
            if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
                self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])
                self._soft_update_target(self.critics_target[agent_id], self.critics[agent_id])

    
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    def choose_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = np.zeros((self.n_agents, self.action_dim))

        for agent_id in range(self.n_agents):
            action_var = (self.actors[agent_id](state_var[:,agent_id,:]))
            if self.use_cuda:
                action[agent_id] = action_var.data.cpu().numpy()[0]
            else:
                action[agent_id] = action_var.data.numpy()[0]
        
        for n in range(self.n_agents):
            for i in range(6):
                if (self.n_episodes < 600): e = self.n_episodes
                else: e = self.n_episodes
                action[n][i] = -exp(-e/self.tau) + self.noise     
        b = 1
        a = -1
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


    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents*self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents*self.action_dim)
        values = np.zeros(self.n_agents)
        for agent_id in range(self.n_agents):
            value_var = self.critics[agent_id](whole_state_var, whole_action_var)
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

