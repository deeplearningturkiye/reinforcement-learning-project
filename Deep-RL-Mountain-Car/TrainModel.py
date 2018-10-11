import json
import numpy as np
import matplotlib.pyplot as plt

from MountainCar import MountainCar
from QLearning import QLearning
from sarsa import Sarsa
from double_q import DQLearning
from expected_sarsa import ExpectedSarsa
import random

if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 5
    max_memory = 50000
    batch_size = 32
    input_size = 2

    Xrange = [-1.5, 0.55]
    Vrange = [-2.0, 2.0]
    start = [np.random.randint(7) * 0.1 - 0.5, 0.0]
    goal = [0.45]

    # n_step = 1

    GAMMA = 0.99  # decay rate of past observations
    OBSERVATION = 3200.  # timesteps to observe before training
    EXPLORE = 10000.  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    INITIAL_EPSILON = 0.2  # starting value of epsilon
    LEARNING_RATE = 1e-4
    FRAME_PER_ACTION = 1

    # all possible steps
    nSteps = np.arange(2, 5, 1)

    # all possible alphas
    #alphas = np.arange(0.01, 0.2, 0.1)

    alphas = [0.01]
    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Define environment/game
    env = MountainCar(start, goal, Xrange, Vrange)

    # Initialize experience replay object
    # learning_model = QLearning(num_actions=num_actions, max_memory=max_memory)
    #learning_model = DQLearning(num_actions=num_actions, max_memory=max_memory, e_greedy=INITIAL_EPSILON)
    learning_model = ExpectedSarsa(num_actions=num_actions, max_memory=max_memory, e_greedy=INITIAL_EPSILON)

    for e in range(epoch):
        # loss = 0.
        env = MountainCar(start, goal, Xrange, Vrange)
        env.reset()
        game_over = False

        # get initial input
        s = env.observe()

        t = 0
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

        while not game_over:
            loss = 0
            t += 1

            action = learning_model.get_action(s)

            # We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                learning_model.change_epsilon(epsilon)

            next_state, reward, game_over = env.act(action)

            # store experience
            learning_model.remember((s, action, reward, next_state), game_over)

            # only train if done observing
            if t > OBSERVE:
                loss += learning_model.train(current_step=t, batch_size=batch_size)

            s = next_state

            # save progress every 10000 iterations
            if t % 1000 == 0:
                learning_model.save_model()

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif OBSERVE < t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("Epoch", e, "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/", "REWARD", reward,
                  "/ ACTION ", action, "/ POS", next_state[0, 0], "/ Loss ", loss)

            if t > 20000:   # stop sampling, continue with new episode
                break

        print("Episode finished!")
        print("************************")
        learning_model.save_model()
        training = False

