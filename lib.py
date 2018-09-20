"""
Reinforcement learning library

Included methods:
    Temporal Difference
    Q Learning
    SARSA
    Double Q Learning
"""
import numpy as np
import gym
import random
from collections import defaultdict


class policy():
    def __init__(self, epsilon, action_space):
        
        self.epsilon=epsilon
        self.nA=action_space

    def probs(self,q_table,observation):
        A_probs = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(q_table[observation])
        A_probs[best_action] += (1 - self.epsilon)

        return A_probs


class q_learning_agent():
    def __init__(self, epsilon, discount_factor, alpha, action_space):
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def select_action(self,state):
        A_probs = self.policy.probs(self.q_table,state)

        return np.random.choice(np.arange(len(A_probs)), p=A_probs)

    def get_q_table(self):
        return self.q_table

    def set_q_table(self, q_table):
        self.q_table = q_table


class sarsa_learning_agent():
    def __init__(self, epsilon, discount_factor, alpha, action_space):
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        next_action = self.select_action(next_state)
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def select_action(self,state):
        A_probs = self.policy.probs(self.q_table,state)

        return np.random.choice(np.arange(len(A_probs)), p=A_probs)

    def get_q_table(self):
        return self.q_table

    def set_q_table(self, q_table):
        self.q_table = q_table

class double_q_learning_agent():
    def __init__(self, epsilon, discount_factor, alpha, action_space):
        self.q_table_1 = defaultdict(lambda: np.zeros(action_space))
        self.q_table_2 = defaultdict(lambda: np.zeros(action_space))
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.alpha = alpha
        self.policy = policy(self.epsilon, self.action_space)

    def learn(self, action, reward, state, next_state):
        if random.randint(1, 2) == 1:
            next_action = np.argmax(self.q_table_1[next_state])
            td_target = reward + self.discount_factor * self.q_table_2[next_state][next_action]
            td_error = td_target - self.q_table_1[state][action]
            self.q_table_1[state][action] += self.alpha * td_error
        else:
            next_action = np.argmax(self.q_table_2[next_state])
            td_target = reward + self.discount_factor * self.q_table_1[next_state][next_action]
            td_error = td_target - self.q_table_2[state][action]
            self.q_table_2[state][action] += self.alpha * td_error

    def select_action(self, state):
        A_probs_1 = self.policy.probs(self.q_table_2, state)
        A_probs_2 = self.policy.probs(self.q_table_1, state)

        return np.random.choice(np.arange(len(A_probs_1)), p=(A_probs_1+A_probs_2)/2)

    def get_q_tables(self):
        return [self.q_table_1, self.q_table_2]

    def set_q_table(self, q_table_1, q_table_2):
        self.q_table_1 = q_table_1
        self.q_table_2 = q_table_2
