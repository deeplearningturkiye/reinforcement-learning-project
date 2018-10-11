class Agent(object):
    def __init__(self, max_memory=100, discount=.99):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def __create_model(self, load_model):
        pass

    def episode(self, env, batch_size=10, n_step=1, epoch=0):
        pass

    def save_model(self):
        pass

    def train(self, current_step, batch_size=10, n_step=1):
        pass

    def get_batch(self, batch_size=10, n_step=1):
        pass

    def get_action(self, state):
        pass

