import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from Agent import Agent


class QLearning(Agent):

    def __init__(self, num_actions, max_memory=100, discount=.99, e_greedy=.1):
        super().__init__(max_memory, discount)
        self.num_actions = num_actions
        self.epsilon = e_greedy

        self.__create_model()

    def __create_model(self, load_model=True):
        hidden_size = 100

        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape=(2,), activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=0.01), "mse")

    def get_batch(self, batch_size=10, n_step=1):
        len_memory = len(self.memory)-n_step    # we don't want to update 'n' last states, because their returns not calculated yet
        num_actions = self.model.output_shape[-1]

        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = self.model.predict(state)[0]
            Q_sa = np.max(self.model.predict(next_state)[0])

            if game_over:  # if game_over is True
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.num_actions, size=1)[0]
        else:
            return np.argmax(self.model.predict(state)[0])

    def get_model(self):
        return self.model
