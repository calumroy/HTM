
from reinforcement_learning import Thalamus
import numpy as np


class test_Thalamus:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3
        self.newThalamus = Thalamus(self.width, self.height)

        self.input1 = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        self.input2 = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        self.combinedInput = np.append([self.input1], [self.input1], axis=0)
