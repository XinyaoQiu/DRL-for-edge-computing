
import random
from collections import namedtuple


Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "next_actions", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, next_action=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, next_action, done)
        self.position = (self.position + 1) % self.capacity

    # def push(self, states, actions, rewards, next_states=None, dones=None):
        
    
    #     # print("states = ", states)
    #     # print("actions = ", actions)
    #     # print("rewards = ", rewards)
    #     # print("next_states = ", next_states)
    
    #     # print("dones = ", dones)
    #     if isinstance(states, list):
    #         if next_states is not None and len(next_states) > 0:
    #             self._push_one(states, actions, rewards, next_states, dones)
    #         else:
    #             self._push_one(states, actions, rewards)
    #     else:
    #         self._push_one(states, actions, rewards, next_states, dones)

    def push(self, states, actions, rewards, next_states=None, next_actions=None, dones=None):

        if isinstance(states, list):
            if dones is not None and len(next_states) > 0:
                for s,a,r,n_s, n_a, d in zip(states, actions, rewards, next_states, next_actions, dones):
                    self._push_one(s, a, r, n_s, n_a, d)
            elif next_states is not None:
                for s,a,r, n_s, n_a in zip(states, actions, rewards, next_states, next_actions):
                    self._push_one(s, a, r, n_s, n_a)
            else:
                for s,a,r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, next_actions, dones)


    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
