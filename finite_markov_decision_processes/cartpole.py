import pprint
import numpy as np
import gym


# cart_position_bins = np.linspace(-2.4, 2.4, 10)
# cart_velocity_bins = np.linspace(-1, 1, 10)
  
class cartPole:
  def __init__(self, epsilon, gamma):
    self.qTable = {}
    self.epsilon = epsilon
    self.gamma = gamma
    
    self.env = gym.make('CartPole-v0')
    self.actions = [0, 1]

    self.pole_angle_bins = np.linspace(-0.42, 0.42, 10)
    self.pole_velocity_bins = np.linspace(-1, 1, 10)

    self.timesteps_over_time = []

  def observationToState(self, observation):
    pole_angle = np.digitize(x=[observation[2]], bins=self.pole_angle_bins)[0]
    pole_velocity = np.digitize(x=[observation[3]], bins=self.pole_velocity_bins)[0]

    return (pole_angle, pole_velocity)

  def updateQTable(self, state, action, reward):
    currentValue = self.qTable.get((state, action), None)

    if currentValue is None:
      self.qTable[(state, action)] = reward
    else:
      self.qTable[(state, action)] = currentValue + self.gamma * (reward - currentValue)

  def chooseAction(self, state):
    if np.random.random() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      q = [self.qTable.get((state, action), 0.0) for action in self.actions]
      action = self.actions[np.argmax(q)]

    return action

  def run(self):
    for i_episode in range(1000):
      observation = self.env.reset()
      state = self.observationToState(observation)

      done = False
      ts = 1

      episodeStates = []
      episodeActions = []
      

      while not done:
        self.env.render()

        action = self.chooseAction(state)

        observation, reward, done, info = self.env.step(action)

        episodeStates.append(state)
        episodeActions.append(action)

        state = self.observationToState(observation)

        if done:
          print("Episode {} finished after {} timesteps".format(i_episode ,ts))

        ts += 1

      for i in range(len(episodeStates)):
        state = episodeStates[i]
        action = episodeActions[i]

        self.updateQTable(state, action, ts)

      self.timesteps_over_time.append(ts)

cp = cartPole(0.1, 0.5)
cp.run()

pp = pprint.PrettyPrinter(depth=6)
pp.pprint(cp.qTable)

pp.pprint(cp.timesteps_over_time)