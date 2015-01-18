from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM, HTMLayer, HTMRegion, Column
import numpy as np
import GUI_HTM


class simpleVerticalLineInputs:
    def __init__(self):
        self.inputs = np.array([[[0 for i in range(self.width * self.cellsPerColumn)]
                                for j in range(self.height)] for k in range(self.numInputs)])
        self.setInputs()
        # Use an index to keep track of which input to send next
        self.index = 0

    def setInputs(self, inputs):
        # Will will create vertical lines in the input that move from side to side.
        # These inputs should then be used to test temporal pooling.
        for n in range(len(inputs)):
            for y in range(len(inputs[0])):
                for x in range(len(inputs[n][y])):
                    if x == n:
                        inputs[n][y][x] = 1

    def createSimGrid(self):
        newGrid = self.inputs[self.index]
        self.index += 1
        if (self.index > len(self.inputs)):
            self.index = 0
        return newGrid


class test_TemporalPooling:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3
        self.numLevels = 1

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        self.numInputs = 10

        self.setInputs(self.inputs)
        self.InputCreator = simpleVerticalLineInputs()
        #self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)
        self.htm = HTM_V.HTM(self.numLevels, self.InputCreator.createSimGrid(), self.width, self.height, self.cellsPerColumn)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)



    def test_case1(self):
        pass
