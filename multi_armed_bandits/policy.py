from bandit import bandit
import numpy as np
import matplotlib.pyplot as plt


run = 2000
ts = 1000

variance = 1
limMin = 0
limMax = 3

def greedyPolicy():
	averageRewards = np.zeros(ts)

	for num in range(run):
		bnd = bandit(variance=variance, min=limMin, max=limMax)

		for t in range(ts):
			averageRewards[t] += bnd.takeAction()

	return averageRewards / run

def epsilonGreedyPolicy(epsilon):
	averageRewards = np.zeros(ts)

	for num in range(run):
		bnd = bandit(variance=variance, min=limMin, max=limMax, epsilon=epsilon)

		for t in range(ts):
			averageRewards[t] += bnd.takeAction()

	return averageRewards / run

def optimisticInitialValues(initial):
	averageRewards = np.zeros(ts)

	for num in range(run):
		bnd = bandit(variance=variance, min=limMin, max=limMax, initial=initial)

		for t in range(ts):
			averageRewards[t] += bnd.takeAction()

	return averageRewards / run

plt.plot(greedyPolicy(), color='r')
plt.plot(epsilonGreedyPolicy(0.1), color='b')
plt.plot(epsilonGreedyPolicy(0.01), color='g')
# plt.plot(optimisticInitialValues(5), color='g')


plt.show()
