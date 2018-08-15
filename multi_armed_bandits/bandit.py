import math
import numpy as np


class bandit:
  def __init__(self, kArm=10, epsilon=0, initial=0., variance=1, min=-2, max=2):
    self.epsilon = epsilon
    self.kArm = kArm
    self.variance = variance

    self.step = 0
    self.qTable = np.full(self.kArm, initial)
    self.armMeans = np.random.uniform(min, max, self.kArm)

  def takeAction(self):
    if self.epsilon > 0 and np.random.binomial(1, self.epsilon) == 1:
      idx = np.random.choice(self.kArm)
    else:
      idx = np.argmax(self.qTable)

    reward = np.random.normal(self.armMeans[idx], self.variance)

    self.step += 1
    self.qTable[idx] +=  (1 / self.step) * (reward - self.qTable[idx])

    return reward
