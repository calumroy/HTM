from mock import MagicMock
from mock import patch
from Inverted_Pendulum import InvertedPendulum
import numpy as np


class TestExampleTwo:
    def test_c(self):
        assert 'c' == 'c'


class test_InvertedPendulum:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.invertedPendulum = InvertedPendulum(self.width, self.height)
        self.input = np.array([[0 for i in range(self.width)] for j in range(self.height)])

    #@patch('HTM_Balancer.Column.predictiveStateArray')
    def test_convertSDRtoAcc_min(self):
        minAcc = -1
        maxAcc = 1
        self.invertedPendulum.minAcc = minAcc
        self.invertedPendulum.maxAcc = maxAcc
        for y in range(len(self.input)):
            for x in range(len(self.input[0])):
                if x == 0:
                    self.input[y][x] = 1
        assert self.invertedPendulum.minAcc == minAcc
        assert self.invertedPendulum.convertSDRtoAcc(self.input) == self.invertedPendulum.minAcc

    def test_convertSDRtoAcc_med(self):
        minAcc = -1
        maxAcc = 1
        self.invertedPendulum.minAcc = minAcc
        self.invertedPendulum.maxAcc = maxAcc
        for y in range(len(self.input)):
            for x in range(len(self.input[0])):
                if x == round(self.width/2):
                    self.input[y][x] = 1
        #import ipdb; ipdb.set_trace()
        assert self.invertedPendulum.convertSDRtoAcc(self.input) == round((minAcc+maxAcc)/2)



