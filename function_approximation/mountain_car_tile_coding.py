import numpy as np
import gym

import sys

from tiles3 import *

env = gym.make('MountainCar-v0')

GAMMA = 1.0
LAMBDA = 0.9

IHT_SIZE = 4096
num_tilings = 8
weights = np.zeros((IHT_SIZE, 1))
z = np.zeros((IHT_SIZE, 1))  # Eligibility trace vector.
iht = IHT(IHT_SIZE)

POSITION_MIN, VELOCITY_MIN = env.env.low
POSITION_MAX, VELOCITY_MAX = env.env.high


def get_active_tiles(state, action):
    pos, vel = state
    active_tiles = tiles(iht, num_tilings, [pos * num_tilings / (POSITION_MAX - POSITION_MIN),
                                            vel * num_tilings / (VELOCITY_MAX - VELOCITY_MIN)],
                         [action])
    return active_tiles


def s_a_feature_vector(state, action):
    active_tiles = get_active_tiles(state, action)
    feature_vector = np.zeros((IHT_SIZE, 1))
    feature_vector[active_tiles] = 1
    return feature_vector


def get_value(state, action):
        # If the state is terminal.
    if state[0] >= POSITION_MAX:
        return 0

    return np.dot(weights.T, s_a_feature_vector(state, action))


def get_action(state):
    values = [get_value(state, action) for action in range(env.action_space.n)]
    return np.argmax(values)


alpha = 0.5
step_size = alpha / num_tilings
n_episodes = 100

# PAGE 305 : Sarsa(λ) with binary features and linear function approximation
for episode in range(n_episodes):
    if episode % 10 == 0:
        print('\rEpisode {}/{}'.format(episode + 1, n_episodes), end='')
        sys.stdout.flush()

    state = env.reset()
    while True:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)

        delta = reward

        active_tiles = get_active_tiles(state, action)
        delta -= get_value(state, action)
        z[active_tiles] = 1

        # If the next state is terminal state.
        if next_state[0] >= POSITION_MAX:
            weights += step_size * delta * z

        next_action = get_action(next_state)
        active_tiles = get_active_tiles(next_state, next_action)
        delta += GAMMA * get_value(next_state, next_action)

        weights += step_size * delta * z
        z = GAMMA * LAMBDA * z

        if done:
            break

        state = next_state

print('Training is done')

# Test the algorithm for 1 episode.
for i in range(1):
    state = env.reset()
    while True:
        env.render()
        action = get_action(state)
        # action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            break

        state = next_state
