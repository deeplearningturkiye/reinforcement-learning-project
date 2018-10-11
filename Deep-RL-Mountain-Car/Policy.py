import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, number_of_action, epsilon=.1, ):
        self.epsilon = epsilon
        self.number_of_action = number_of_action

    def get_action(self, q_values):
        pi = self.get_pi(q=q_values)

        return np.random.choice(np.arange(self.number_of_action), p=pi)
    
    def get_pi(self, q):
        pi = np.ones(self.number_of_action, dtype=float)  # init values
        pi *= self.epsilon / self.number_of_action  # probability of random action

        greedy_action = np.argmax(q)
        pi[greedy_action] += (1.0 - self.epsilon)  # probability of greedy action
        
        return pi
