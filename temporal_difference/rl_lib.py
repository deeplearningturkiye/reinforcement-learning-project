"""
Reinforcement learning kütüphanesi
içerik ==> Temporal difference,Monte Carlo metodları
"""
import numpy as np
import gym
from collections import defaultdict
import random
from gym import spaces


class Policy():
    def __init__(self, epsilon, action_space):
        
        self.epsilon=epsilon
        self.nA=action_space

    def probs(self,q_table,observation):
        A_probs = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(q_table[observation])
        A_probs[best_action] += (1 - self.epsilon)
        return A_probs


class Q_learning_agent():
    def __init__(self, epsilon, discount_factor,alpha, action_space):
        self.Q_table = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = Policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        next_action = np.argmax(self.Q_table[next_state])
        td_target = reward + self.discount_factor * self.Q_table[next_state][next_action]
        td_error = td_target - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error

    def select_action(self,state):
        A_probs = self.policy.probs(self.Q_table,state)
        return np.random.choice(np.arange(len(A_probs)), p=A_probs)

    def return_Q_table(self):
        return self.Q_table


class Sarsa_learning_agent():
    def __init__(self, epsilon, discount_factor,alpha, action_space):
        self.Q_table = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = Policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        next_action = self.select_action(next_state)
        td_target = reward + self.discount_factor * self.Q_table[next_state][next_action]
        td_error = td_target - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error

    def select_action(self,state):
        A_probs = self.policy.probs(self.Q_table,state)
        return np.random.choice(np.arange(len(A_probs)), p=A_probs)

    def return_Q_table(self):
        return self.Q_table

class Double_Q_Learning_Agent():
    def __init__(self, epsilon, discount_factor,alpha, action_space):
        self.Q_table_1 = defaultdict(lambda: np.zeros(action_space))
        self.Q_table_2 = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = Policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        if random.randint(1, 2) == 1:
            next_action = np.argmax(self.Q_table_1[next_state])
            td_target = reward + self.discount_factor * self.Q_table_2[next_state][next_action]
            td_error = td_target - self.Q_table_1[state][action]
            self.Q_table_1[state][action] += self.alpha * td_error
        else:
            next_action = np.argmax(self.Q_table_2[next_state])
            td_target = reward + self.discount_factor * self.Q_table_1[next_state][next_action]
            td_error = td_target - self.Q_table_2[state][action]
            self.Q_table_2[state][action] += self.alpha * td_error

    def select_action(self, state):
        A_probs_1 = self.policy.probs(self.Q_table_2, state)
        A_probs_2 = self.policy.probs(self.Q_table_1, state)
        return np.random.choice(np.arange(len(A_probs_1)), p=(A_probs_1+A_probs_2)/2)

    def return_Q_tables(self):
        return [self.Q_table_1, self.Q_table_2]



