from mock import MagicMock
from mock import patch
import numpy as np
from utilities import sdrFunctions as SDRFunct


class test_sdrFunctions:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3

        self.input1 = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        self.input2 = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        self.combinedInput = np.append([self.input1], [self.input1], axis=0)

    def test_checkArraySizesMatch(self):
        output = SDRFunct.checkArraySizesMatch(self.input1, self.combinedInput[0])
        assert output is True

    def test_checkArraySizesDontMatch(self):
        output = SDRFunct.checkArraySizesMatch(self.input1, self.input2)
        assert output is False

    def test_rangeOfSizesMatch(self):
        for i in range(0, 100, 5):
            self.input1 = np.array([[0 for m in range(i)] for n in range(self.height)])
            self.combinedInput = np.append([self.input1], [self.input1], axis=0)
            output = SDRFunct.checkArraySizesMatch(self.input1, self.combinedInput[1])
            output2 = SDRFunct.checkArraySizesMatch(self.input1, self.input1)
            assert output is True
            assert output2 is True
