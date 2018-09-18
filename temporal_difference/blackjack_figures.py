
# REFERENCE : https://github.com/dennybritz/reinforcement-learning

import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
from collections import defaultdict

env = gym.make('Blackjack-v0')


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A_probs[best_action] += (1 - epsilon)
        return A_probs

    return policy_fn


def sarsa(env, n_episodes=500, discount_factor=1.0, alpha=0.5, epsilon=0.1):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(n_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, n_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        while True:
            A_probs = policy(state)
            action = np.random.choice(np.arange(len(A_probs)), p=A_probs)
            next_state, reward, done, _ = env.step(action)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            if done:
                break

            state = next_state

    return Q


def Q_learning(env, n_episodes=500, discount_factor=1.0, alpha=0.5, epsilon=0.1):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(n_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, n_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            if done:
                break

            state = next_state

    return Q


def plot_figure(ax, usable_ace):
    def get_action(player_hand, dealer_showing, usable_ace):
        return policy[player_hand, dealer_showing, usable_ace] if (player_hand, dealer_showing, usable_ace) in policy else 1

    policy_mat = np.array([[get_action(player_hand, dealer_showing, usable_ace) for dealer_showing in range(1, 11)]
                           for player_hand in range(21, 10, -1)])

    ax.imshow(policy_mat, cmap=plt.cm.Accent, extent=[0.5, 10.5, 10.5, 21.5])
    plt.ylim(11, 21)
    plt.xlim(1, 10)
    plt.xlabel('Dealer Hand')
    plt.ylabel('Player Hand')
    hit_patch = mpatches.Patch(color=plt.cm.Accent(.1), label='Stick')
    stick_patch = mpatches.Patch(color=plt.cm.Accent(.9), label='Hit')
    plt.legend(handles=[hit_patch, stick_patch])


# Q = sarsa(env, 1000)
Q = Q_learning(env, 1000)
policy = dict((k, np.argmax(v)) for k, v in Q.items())

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(121)
ax.set_title('Blackjack MC Policy - No Usable Ace')
plot_figure(ax, True)
ax = fig.add_subplot(122)
ax.set_title('Blackjack MC Policy - Usable Ace')
plot_figure(ax, False)
plt.show()
