from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM, HTMLayer, HTMRegion, Column
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import random
from copy import deepcopy


class simpleVerticalLineInputs:
    def __init__(self, width, height, numInputs):
        # The number of inputs to store
        self.numInputs = numInputs
        self.width = width
        self.height = height
        self.inputs = np.array([[[0 for i in range(self.width)]
                                for j in range(self.height)] for k in range(self.numInputs)])
        self.setInputs(self.inputs)
        # Use an index to keep track of which input to send next
        self.index = 0
        # A variable speifying the amount of noise in the inputs 0 to 1
        self.noise = 0.0

    def setInputs(self, inputs):
        # Will will create vertical lines in the input that move from side to side.
        # These inputs should then be used to test temporal pooling.
        for n in range(len(inputs)):
            for y in range(len(inputs[0])):
                for x in range(len(inputs[n][y])):
                    if x == n:
                        inputs[n][y][x] = 1

    def step(self, cellGrid):
        # Required function for a InputCreator class
        pass

    def createSimGrid(self):
        # Required function for a InputCreator class
        # Return a new grid
        newGrid = deepcopy(self.inputs[self.index])

        # Add some random noise to the next input
        # The next input is at the self.index
        if self.noise > 0.0:
            for y in range(len(newGrid[0])):
                for x in range(len(newGrid[y])):
                    if random.random() < self.noise:
                        newGrid[y][x] = 1

        self.index += 1
        if (self.index >= len(self.inputs)):
            self.index = 0
        return newGrid


class test_TemporalPooling:
    def setUp(self):
        self.width = 10
        self.height = 30
        self.cellsPerColumn = 3
        self.numLevels = 2

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = self.width*self.cellsPerColumn
        inputWidth = self.width*self.cellsPerColumn
        inputHeight = 2*self.height

        self.InputCreator = simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)
        #self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)
        self.htm = HTM(self.numLevels, self.InputCreator.createSimGrid(), self.width, self.height, self.cellsPerColumn)

        # Setup some parameters of the HTM
        self.setupParameters()

        app = QtGui.QApplication(sys.argv)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        sys.exit(app.exec_())

    def setupParameters(self):
        # Setup some parameters of the HTM
        self.htm.regionArray[0].layerArray[1].desiredLocalActivity = 4
        self.htm.regionArray[1].layerArray[0].desiredLocalActivity = 4
        self.htm.regionArray[1].layerArray[1].desiredLocalActivity = 4

    def test_case1(self):
        pass
