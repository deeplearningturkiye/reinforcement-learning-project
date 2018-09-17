import gym
import numpy as np
import random


env = gym.make('Blackjack-v0')
env.reset()


def make_epsilon_greedy_policy(Q_1, Q_2, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A_probs = np.ones(nA, dtype=float) * epsilon / nA

        Q = Q_1 + Q_2

        best_action = np.argmax(Q[observation[0]][observation[1]][int(observation[2])])
        A_probs[best_action] += (1.0 - epsilon)

        return A_probs  # .reshape(1,-1)

    return policy_fn


# Q Learning
def double_Q_learning(env, train_episodes, test_episodes, discount_factor=0.2, alpha=0.5, epsilon=0.1):
    # Q table initialization
    Q_1 = np.zeros((32, 11, 2, env.action_space.n))
    Q_2 = np.zeros((32, 11, 2, env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q_1, Q_2, epsilon, env.action_space.n)

    for i_episode in range(train_episodes):
        # Reset the environment and pick the first action
        observation = env.reset()
        state = observation

        while True:
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_observation, reward, done, _ = env.step(action)
            next_state = next_observation

            if(random.randint(1, 2) == 1):
                # Q Learning Update
                best_next_action = np.argmax(Q_1[next_state[0]][next_state[1]][int(next_state[2])])
                td_target = reward + discount_factor * Q_2[next_state[0]][next_state[1]][int(next_state[2])][best_next_action]
                td_delta = td_target - Q_1[state[0]][state[1]][int(state[2])][action]
                Q_1[state[0]][state[1]][int(state[2])][action] += alpha * td_delta

            else:
                # Q Learning Update
                best_next_action = np.argmax(Q_2[next_state[0]][next_state[1]][int(next_state[2])])
                td_target = reward + discount_factor * Q_1[next_state[0]][next_state[1]][int(next_state[2])][best_next_action]
                td_delta = td_target - Q_2[state[0]][state[1]][int(state[2])][action]
                Q_2[state[0]][state[1]][int(state[2])][action] += alpha * td_delta

            state = next_state

            if done:
                break

    policy = make_epsilon_greedy_policy(Q_1, Q_2, 0, env.action_space.n)

    win_count = 0
    reward_sum = 0

    for i_episode in range(test_episodes):
        # Reset the environment and pick the first action
        observation = env.reset()
        state = observation

        while True:
            # Take a step
            action_probs = policy(observation)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            observation, reward, done, _ = env.step(action)

            reward_sum += reward

            if reward > 0:
                win_count += 1

            if done:
                break

    print("Total Episodes: {}".format(test_episodes))
    print("Win Count: {}".format(win_count))
    print("Reward Sum: {}".format(reward_sum))


double_Q_learning(env, 10000, 1000)
