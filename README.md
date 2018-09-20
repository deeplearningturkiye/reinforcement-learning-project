# Deep Learning Türkiye - Reinforcement Learning Project

This repository consists projects from Deep Learning Türkiye - Reinforcement Learning Group. Enter folders to see each project's details.

## 1. Introduction To RL
Simple tic tac toe example. Learns via Value Function at the moment. Policy Search *TODO*.
Benefited from [tansey](https://github.com/tansey/rl-tictactoe/blob/master/tictactoe.py).

## 2. Multi-Armed Bandits
Provides the underlying testbed for bandit problem.

## 3. Finite Markov Decision Processes
Uses the OpenAI Gym. Learns via Q-Learning.

## 4. Temporal Difference
Multiple approaches to CartPole problem.
Benefited from [dennybritz](https://github.com/dennybritz/reinforcement-learning).

## Library usage
You can find example usage below.

```
import gym
from lib import q_learning_agent, double_q_learning_agent, sarsa_learning_agent

env = gym.make("FrozenLake-v0")
env.reset()

def train(agent):
    for i_episode in range(1000):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(action, reward, state, next_state)
            if done:
                break
            state = next_state

qla = q_learning_agent(epsilon=0.3, discount_factor=0.9, alpha=0.5, action_space=env.action_space.n)
sla = sarsa_learning_agent(epsilon=0.3, discount_factor=0.9, alpha=0.5, action_space=env.action_space.n)
dqla = double_q_learning_agent(epsilon=0.3, discount_factor=0.9, alpha=0.5, action_space=env.action_space.n)

train(qla)
train(sla)
train(dqla)
```