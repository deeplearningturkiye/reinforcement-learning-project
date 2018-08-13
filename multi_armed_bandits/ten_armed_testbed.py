import math
from random import gauss, uniform


class ten_armed_testbed:
    def __init__(self, variance=1, min=-2, max=2):
        self.means = [uniform(min, max) for i in range(10)]
        self.variance = variance
    def pullArm(self, k):
        return gauss(self.means[k], math.sqrt(self.variance))

