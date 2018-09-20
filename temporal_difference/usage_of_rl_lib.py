import gym
from rl_lib import Q_learning_agent, Double_Q_Learning_Agent , Sarsa_learning_agent


env = gym.make("FrozenLake-v0")
env.reset()

Agent = Q_learning_agent(epsilon=0.3, discount_factor=0.9,alpha=0.5,action_space=env.action_space.n)
n_episodes = 1000
total_reward_for_q = 0
for i_episode in range(n_episodes):
        state = env.reset()
        while True:
            action = Agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            Agent.learn(action, reward, state, next_state)
            total_reward_for_q += reward
            if done:
                break
            state = next_state
# print(len(Agent.return_Q_table()), env.observation_space.n)
print("total reward Q learning  : ", total_reward_for_q)
env.reset()

total_reward_sarsa = 0
Agent = Sarsa_learning_agent(epsilon=0.3, discount_factor=0.9, alpha=0.5, action_space=env.action_space.n)
n_episodes = 1000
for i_episode in range(n_episodes):
        state = env.reset()
        while True:
            action = Agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            Agent.learn(action, reward, state, next_state)
            total_reward_sarsa += reward
            if done:
                break
            state = next_state
print("total reward sarsa  : ", total_reward_sarsa)
# print(Agent.return_Q_table())


total_reward_double_q_learning = 0
Agent = Double_Q_Learning_Agent(epsilon=0.3, discount_factor=0.9, alpha=0.5, action_space=env.action_space.n)
n_episodes = 1000
for i_episode in range(n_episodes):
        state = env.reset()
        while True:
            action = Agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            Agent.learn(action, reward, state, next_state)
            total_reward_double_q_learning += reward
            if done:
                break
            state = next_state
print("total reward double Q  learning  : ", total_reward_double_q_learning)
# print(Agent.return_Q_table())
