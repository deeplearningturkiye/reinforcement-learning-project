import random
import numpy as np
import matplotlib.pyplot as plt

from MountainCar import MountainCar
from sarsa import Sarsa
from double_q import DQLearning
from expected_sarsa import ExpectedSarsa


def test_double_q():
    runs = 1
    episode = 2

    alphas = []
    steps = np.zeros((len(alphas), episode))

    for run in range(0, runs):
        for i, n_step in zip(range(len(alphas)), alphas):
            learning_model = DQLearning(num_actions=num_actions, max_memory=max_memory, e_greedy=epsilon, load_model=False)

            for e in range(episode):
                print('run:', run, 'alpha:', alphas[i], 'episode:', e)

                start = [random.uniform(-0.6, -0.4), 0.0]
                env = MountainCar(start, goal, Xrange, Vrange)

                step = learning_model.episode(env=env, batch_size=batch_size, n_step=n_step, epoch=e)
                steps[i, e] += step

    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = ' + str(alphas[i]))
    plt.xlabel('Alpha')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.show()


def one_vs_multi_step():
    runs = 1
    episode = 10
    n_steps = np.arange(1, 9, 1)

    steps = np.zeros((len(n_steps), episode))

    for run in range(0, runs):
        for i, n_step in zip(range(len(n_steps)), n_steps):
            learning_model = ExpectedSarsa(num_actions=num_actions, max_memory=max_memory, e_greedy=epsilon, load_model=False)
            #learning_model = Sarsa(num_actions=num_actions, max_memory=max_memory, e_greedy=epsilon, load_model=False)

            for e in range(episode):
                print('run:', run, 'steps:', n_steps[i], 'episode:', e)

                start = [random.uniform(-0.6, -0.4), 0.0]
                env = MountainCar(start, goal, Xrange, Vrange)

                step = learning_model.episode(env=env, batch_size=batch_size, n_step=n_step, epoch=e)
                steps[i, e] += step

    steps /= runs

    for i in range(0, len(n_steps)):
        plt.plot(steps[i], label='n = '+str(n_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.show()


def effect_of_alpha_and_n():
    # all possible alphas
    alphas = [0.0002, 0.0004, 0.0008, 0.0012]

    # all possible steps
    n_steps = np.arange(1, 9, 1)

    epoch = 20
    runs = 1

    steps = np.zeros((len(n_steps), len(alphas)))

    for run in range(0, runs):
        for nStepIndex, n_step in zip(range(0, len(n_steps)), n_steps):
            for alphaIndex, alpha in zip(range(0, len(alphas)), alphas):
                learning_model = ExpectedSarsa(num_actions=num_actions, max_memory=max_memory, e_greedy=epsilon, load_model=False)
                #learning_model = Sarsa(num_actions=num_actions, max_memory=max_memory, e_greedy=epsilon,
                #                       load_model=False)

                for e in range(0, epoch):
                    print('run:', run, 'steps:', n_step, 'alpha:', alpha, 'episode:', e)

                    start = [random.uniform(-0.6, -0.4), 0.0]
                    env = MountainCar(start, goal, Xrange, Vrange)

                    step = learning_model.episode(env=env, batch_size=batch_size, n_step=n_step, epoch=e)
                    steps[nStepIndex, alphaIndex] += step

    # average over independent runs and episodes
    steps /= runs * epoch

    for i in range(0, len(n_steps)):
        plt.plot(alphas, steps[i, :], label='n = '+str(n_steps[i]))
    plt.xlabel('Alpha')
    plt.ylabel('Steps per episode')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    max_memory = 20000
    batch_size = 100
    input_size = 2

    Xrange = [-1.2, 0.6]
    Vrange = [-0.07, 0.07]
    goal = [0.5]

    one_vs_multi_step()
    effect_of_alpha_and_n()
