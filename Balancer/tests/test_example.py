from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM, HTMLayer, HTMRegion, Column
import numpy as np


class TestExampleTwo:
    def test_c(self):
        assert 'c' == 'c'


class test_HTMLayer:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3
        self.input = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        self.htmlayer = HTMLayer(self.input, self.width, self.height, self.cellsPerColumn)

        self.historyLength = 2
        self.predictiveStateArray = np.array([0 for i in range(self.historyLength)])
        for i in range(self.cellsPerColumn-1):   # Minus one since the first entry is already there
            self.predictiveStateArray = np.vstack((self.predictiveStateArray, [0 for i in range(self.historyLength)]))

    #@patch('HTM_Balancer.Column.predictiveStateArray')
    def test_predictiveState(self):
        column = self.htmlayer.columns[0][0]
        column.predictiveStateArray = np.array([[1,0],[3,0],[5,6]])
        #instance = MagicMock.return_value
        #instance.predictiveStateArray.return_value = self.predictiveStateArray

        #column.predictiveStateArray = MagicMock(return_value = np.array([[1,0],[3,0],[5,6]]))

        #import ipdb; ipdb.set_trace()
        result = self.htmlayer.predictiveState(column, 0, 0)
        assert result is True

    #@patch.object(HTM_Balancer.Column.predictiveStateArray,)
    #def test_getBestMatchingCell(self):
    #    timestep = 0
    #    column = Column(3, 0, 0)
    #    self.htmlayer.getBestMatchingCell(column, timeStep)


class test_HTM:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3
        self.numLevels = 1
        self.input = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        self.htm = HTM(self.numLevels, self.input, self.width, self.height, self.cellsPerColumn)

    def test_joinInputArrays(self):
        input1 = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        input2 = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        output = self.htm.joinInputArrays(input1, input2)
        assert len(output) == len(input1) + len(input2)
        assert len(output[0]) == self.width * self.cellsPerColumn

    def test_joinNullInputArrays(self):
        input1 = np.array([])
        input2 = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        output = self.htm.joinInputArrays(input1, input2)
        #import ipdb; ipdb.set_trace()
        assert len(output) == len(input1) + len(input2)
        assert len(output[0]) == len(input2[0])
