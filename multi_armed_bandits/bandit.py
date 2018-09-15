import math
import numpy as np

##deneme
class bandit:
  def __init__(self, kArm=10, epsilon=0, initial=0., ucb=0., variance=1, min=-2, max=2):
    self.epsilon = epsilon
    self.ucb = ucb
    self.kArm = kArm
    self.variance = variance

    self.selectActions = np.zeros(self.kArm)
    self.totalTS = 0
    self.steps = np.zeros(self.kArm)
    self.qTable = np.full(self.kArm, initial)
    self.armMeans = np.random.uniform(min, max, self.kArm)

  def takeAction(self):
    if self.epsilon > 0 and np.random.binomial(1, self.epsilon) == 1:
      idx = np.random.choice(self.kArm)
    elif self.ucb > 0:
      actions = []
      for idx in np.arange(self.kArm):
        if self.selectActions[idx] == 0:
          actions.append(1000)
          break
        else:
          actions.append(self.qTable[idx] + self.ucb * math.sqrt(math.log(self.totalTS) / self.selectActions[idx]))

      idx = np.argmax(np.asarray(actions))
      self.selectActions[idx] += 1 
    else:
      idx = np.argmax(self.qTable)

    reward = np.random.normal(self.armMeans[idx], self.variance)

    self.totalTS += 1
    self.steps[idx] += 1
    self.qTable[idx] += (1 / self.steps[idx]) * (reward - self.qTable[idx])

    return reward
