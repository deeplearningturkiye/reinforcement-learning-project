import numpy as np
import os.path
import json

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras.regularizers import l1

from Agent import Agent
from Policy import EpsilonGreedyPolicy


class DQLearning(Agent):
    def __init__(self, num_actions, max_memory=100, discount=.99, e_greedy=.1, load_model=True):
        super().__init__(max_memory, discount)
        self.num_actions = num_actions
        self.epsilon = e_greedy
        self.policy = EpsilonGreedyPolicy(number_of_action=num_actions, epsilon=e_greedy)

        self.__create_model(load_model)
        self.current_model = self.model

    def change_epsilon(self, epsilon):
        self.policy.epsilon = epsilon

    def episode(self, env, batch_size=10, n_step=1, epoch=0):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        state = env.observe()

        step = 0
        while not game_over:
            # go to next time n_step
            step += 1
            action = self.get_action(state)
            # apply action, get rewards and new state
            next_state, reward, game_over = env.act(action)

            # store experience
            self.remember((state, action, reward, next_state), game_over)

            loss += self.train(current_step=step, batch_size=batch_size, n_step=n_step)
            print('Step {}| epoch {} | n_step {} | Loss {:.4f} |Pos {:.3f} | Act {}'.format(
                step, epoch, n_step, loss, next_state[0, 0], action - 1))

            if np.math.isnan(loss) or step > 1500:
                break

            state = next_state

        print("Episode finished!")
        print("************************")

        return step

    def train(self, current_step, batch_size=10, n_step=1):
        loss = 0.

        model = self.get_model()

        inputs, targets = self.get_batch(batch_size=batch_size, n_step=n_step)

        loss += model.train_on_batch(inputs, targets)

        return loss

    def get_batch(self, batch_size=10, n_step=1):
        len_memory = len(self.memory)
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
                q1 = self.model.predict(next_state)[0]
                q2 = self.model2.predict(next_state)[0]

                if self.current_model == self.model:
                    best_action = np.argmax(q1)
                    q_sa = q2[best_action]
                else:
                    best_action = np.argmax(q2)
                    q_sa = q1[best_action]

                targets[i, action] = reward + self.discount * q_sa

        return inputs, targets

    def get_action(self, state):
        q1 = self.model.predict(state)[0]
        q2 = self.model2.predict(state)[0]

        return self.policy.get_action(q_values=q1+q2)

    def get_model(self):
        aa = np.random.randint(2, size=1)
        if aa == 0:
            self.current_model = self.model
        else:
            self.current_model = self.model2

        return self.current_model

    def __create_model(self, load_model=True):
        hidden_size = 100

        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape=(2,), activation='relu', kernel_regularizer=l1(0.01)))
        self.model.add(Dense(hidden_size, activation='relu', kernel_regularizer=l1(0.01)))
        # self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=0.0001), "mse")

        if load_model and os.path.exists("model.dqlearning1"):
            self.model.load_weights("model.dqlearning1")

        self.model2 = Sequential()
        self.model2.add(Dense(hidden_size, input_shape=(2,), activation='relu', kernel_regularizer=l1(0.01)))
        self.model2.add(Dense(hidden_size, activation='relu', kernel_regularizer=l1(0.01)))
        # self.model2.add(Dense(hidden_size, activation='relu'))
        self.model2.add(Dense(self.num_actions))
        self.model2.compile(sgd(lr=0.0001), "mse")

        if load_model and os.path.exists("model.dqlearning2"):
            self.model.load_weights("model.dqlearning2")

    def save_model(self):
        # Save trained model weights and architecture, this will be used by the visualization code
        self.model.save_weights("model.dqlearning1", overwrite=True)
        self.model2.save_weights("model.dqlearning2", overwrite=True)

        with open("model3.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

        with open("model4.json", "w") as outfile:
            json.dump(self.model2.to_json(), outfile)