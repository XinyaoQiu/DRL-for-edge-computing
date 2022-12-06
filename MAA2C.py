from torch import nn
import torch as th
from torch.optim import Adam, RMSprop
from math import exp

import numpy as np
from Memory import ReplayMemory

from Model import ActorNetwork, CriticNetwork
from utils import entropy, to_tensor_var

EVAL_EPISODES = 10

class MAA2C(object):
    """
    An multi-agent learned with Advantage Actor-Critic
    - Actor takes its local observations as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy

    Parameters
    - training_strategy:
        - cocurrent
            - each agent learns its own individual policy which is independent
            - multiple policies are optimized simultaneously
        - centralized (see MADDPG in [1] for details)
            - centralized training and decentralized execution
            - decentralized actor map it's local observations to action using individual policy
            - centralized critic takes both state and action from all agents as input, each actor
                has its own critic for estimating the value function, which allows each actor has
                different reward structure, e.g., cooperative, competitive, mixed task
    - actor_parameter_sharing:
        - True: all actors share a single policy which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous. Please see Sec. 4.3 in [2] and
            Sec. 4.1 & 4.2 in [3] for details.
        - False: each actor use independent policy
    - critic_parameter_sharing:
        - True: all actors share a single critic which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous and reward sharing holds. Please
            see Sec. 4.1 in [3] for details.
        - False: each actor use independent critic (though each critic can take other agents actions
            as input, see MADDPG in [1] for details)

    Reference:
    [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
    [2] Cooperative Multi-Agent Control Using Deep Reinforcement Learning
    [3] Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multi-agent Reinforcement Learning

    """
    def __init__(self, env, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 noise, bound, memory_capacity=10, max_steps=None,
                 roll_out_n_steps=10, tau=300, 
                 reward_gamma=0.99, reward_scale=1., done_penalty=-10,
                 actor_output_act=nn.functional.softmax, critic_loss="huber",
                 actor_lr=0.01, critic_lr=0.01, training_strategy="centralized",
                 optimizer_type="rmsprop", entropy_reg=0.00,
                 max_grad_norm=None, batch_size=10, episodes_before_train=0,
                 use_cuda=False, actor_parameter_sharing=False, critic_parameter_sharing=False, 
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=100):

        
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound
        self.noise = noise
        self.tau = tau
        self.bound = bound

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

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
        self.target_tau = 0.01

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.n_agents = n_agents
        self.roll_out_n_steps = roll_out_n_steps
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing

        assert training_strategy in ["cocurrent", "centralized"]
        self.training_strategy = training_strategy
        

        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_act)] * self.n_agents
        
        
        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, 1)] * self.n_agents
            
        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        # tricky and memory consumed implementation of parameter sharing
        if self.actor_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.actors[agent_id] = self.actors[0]
                self.actors_optimizer[agent_id] = self.actors_optimizer[0]
        if self.critic_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.critics[agent_id] = self.critics[0]
                self.critics_optimizer[agent_id] = self.critics_optimizer[0]

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
        next_states = []
        next_actions = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.choose_action(self.env_state)
            next_state, reward, done, _, phi, _, _, _ = self.env.step(action)
            next_state_var = to_tensor_var([next_state], self.use_cuda)
            next_action = np.zeros((self.n_agents, self.action_dim))
            for agent_id in range(self.n_agents):
                next_action_var = self.actors[agent_id](next_state_var[:,agent_id,:])
                if self.use_cuda:
                    next_action[agent_id] = next_action_var.data.cpu().numpy()[0]
                else:
                    next_action[agent_id] = next_action_var.data.numpy()[0]
            # done = done[0]
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)

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
            action_log_probs = self.actors[agent_id](states_var[:,agent_id,:])
            entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
            action_log_probs = th.sum(action_log_probs * actions_var[:,agent_id,:], 1)
            values = self.critics[agent_id](whole_states_var, whole_actions_var).detach()
            
            advantages = rewards_var[:,agent_id,:] - values
            pg_loss = -th.mean(action_log_probs * advantages)
            actor_loss = pg_loss - entropy_loss * self.entropy_reg
            actor_loss.requires_grad_(True)
            actor_loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()

            # update critic network
            self.critics_optimizer[agent_id].zero_grad()
            target_values = rewards_var[:,agent_id,:]
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.requires_grad_(True)
            critic_loss.backward()


            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critics_optimizer[agent_id].step()


    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    # predict action based on state for execution
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
                if (self.n_episodes < self.bound): e = self.n_episodes
                else: e = self.bound
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


    # evaluate value
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

