import numpy as np
import json
import math
import os.path

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l1

from Agent import Agent
from Policy import EpsilonGreedyPolicy


class ExpectedSarsa(Agent):
    def __init__(self, num_actions, max_memory=100, discount=.99, e_greedy=.1, load_model=True):
        super().__init__(max_memory, discount)
        self.num_actions = num_actions
        self.policy = EpsilonGreedyPolicy(number_of_action=num_actions, epsilon=e_greedy)
        self.next_action = None

        self.__create_model(load_model)

    def __create_model(self, load_model):
        hidden_size = 100

        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape=(2,), activation='relu', kernel_regularizer=l1(0.01)))
        self.model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l1(0.01)))
        self.model.add(Dropout(rate=.1))
        self.model.add(Dense(self.num_actions))

        sgd = SGD(lr=0.0001, momentum=0.99)
        self.model.compile(optimizer=sgd, loss="mse")

        if load_model and os.path.exists("model.esarsa"):
            self.model.load_weights("model.esarsa")

    def episode(self, env, batch_size=10, n_step=1, epoch=0):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        state = env.observe()

        time = 0
        while not game_over:
            # go to next time n_step
            time += 1

            action = self.get_action(state)

            # apply action, get rewards and new state
            next_state, reward, game_over = env.act(action)

            # store experience
            self.remember((state, action, reward, next_state), game_over)

            loss += self.train(current_step=time, batch_size=batch_size, n_step=n_step)

            print('Step {}| epoch {} | n_step {} | Loss {:.4f} |Pos {:.3f} | Act {}'.format(
                time, epoch, n_step, loss, next_state[0, 0], action - 1))

            if math.isnan(loss) or time > 1500:
                break

            state = next_state

        print("Episode finished!")
        print("************************")

        return time

    def change_epsilon(self, epsilon):
        self.policy.epsilon = epsilon

    def set_learning_rate(self, lr):
        self.model.optimizer.lr = lr

    def save_model(self):
        # Save trained model weights and architecture, this will be used by the visualization code
        self.model.save_weights("model.esarsa", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def train(self, current_step, batch_size=10, n_step=1):
        loss = 0.

        model = self.get_model()

        if current_step - n_step >= 0:
            inputs, targets = self.get_batch(batch_size=batch_size, n_step=n_step)

            loss += model.train_on_batch(inputs, targets)

        return loss

    def get_batch(self, batch_size=10, n_step=1):
        len_memory = len(
            self.memory) - n_step + 1  # we don't want to update 'n' last states,because their returns have not seen yet
        num_actions = self.model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state
            targets[i] = self.model.predict(state)[0]

            if game_over:  # if game_over is True
                targets[i, action] = reward
            else:
                returns = 0.0
                t_n_state = next_state

                for t in range(idx, idx + n_step):
                    _, _, reward, t_n_state = self.memory[t][0]
                    returns += pow(self.discount, t - idx) * reward

                if reward != 100:
                    q = self.model.predict(t_n_state)[0]
                    pi = self.policy.get_pi(q)

                    q_sa = np.dot(q, pi.T)

                    returns += pow(self.discount, n_step) * q_sa

                targets[i, action] = returns

        return inputs, targets

    def get_action(self, state):
        return self.policy.get_action(q_values=self.model.predict(state)[0])

    def get_model(self):
        return self.model
