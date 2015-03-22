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
        # A variable indicating the chance that the next input is a random input from the sequence.
        # This variable is used to create an input sequence that sometimes changes. It is the probablity
        # that the next input is the correct input in the sequence
        self.sequenceProbability = 1.0

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
        newGrid = None
        # Add some random noise to the next input
        # The next input is at the self.index
        if self.noise > 0.0:
            # Return a new grid so the original input is not over written.
            newGrid = deepcopy(self.inputs[self.index])

            for y in range(len(newGrid[0])):
                for x in range(len(newGrid[y])):
                    if random.random() < self.noise:
                        newGrid[y][x] = 1
        # Give the next outpu a chance to be an out of sequence input.
        if (random.random() < self.sequenceProbability):
            outputGrid = self.inputs[self.index]
        else:
            sequenceLen = len(self.inputs)
            outputGrid = self.inputs[random.randint(0, sequenceLen-1)]
        # Increment the index for next time
        self.index += 1
        if (self.index >= len(self.inputs)):
            self.index = 0
        # If noise was added return the noisy grid.
        if newGrid is not None:
            return newGrid
        else:
            return outputGrid


def temporalPoolingPercent(self, grid):
        if self.grid is not None:
            totalPrevActiveIns = np.sum(self.grid != 0)
            totalAndGrids = np.sum(np.logical_and(grid != 0, self.grid != 0))
            percentTemp = float(totalAndGrids) / float(totalPrevActiveIns)
            #print "         totalAndGrids = %s" % totalAndGrids
            self.temporalAverage = (float(percentTemp) +
                                    float(self.temporalAverage*(self.numInputGrids-1)))/float(self.numInputGrids)
            #print "         percentTemp = %s" % percentTemp
        self.grid = deepcopy(grid)
        self.numInputGrids += 1
        return self.temporalAverage



class test_SpatialPooling:
    def setUp(self):
        '''
        We are tesing the spatial pooler.

        A set number of input sequeces are fed into the htm.
        The spatial pooler over time selects certian columns to
        represent certain features of the input. The spatial pooler
        should allow similar inputs to activate similar columns since they contain
        similar features.

        '''
        self.width = 10
        self.height = 30
        self.cellsPerColumn = 3
        self.numLevels = 1
        self.numLayers = 1

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = self.width*self.cellsPerColumn
        inputWidth = self.width*self.cellsPerColumn
        inputHeight = 2*self.height

        self.InputCreator = simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)
        #self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)
        self.htm = HTM(self.numLevels,
                       self.InputCreator.createSimGrid(),
                       self.width,
                       self.height,
                       self.cellsPerColumn,
                       self.numLayers)

        # Setup some parameters of the HTM
        self.setupParameters()

    def setupParameters(self):
        # Setup some parameters of the HTM
        pass

    def step(self):
        # Update the inputs and run them through the HTM levels just once.
        # Update the HTM input and run through the
        newInput = self.InputCreator.createSimGrid()
        self.htm.spatialTemporal(newInput)

    def nSteps(self, numSteps):
        print "Running HTM for %s steps" % numSteps
        for i in range(numSteps):
            self.step()

    def gridsSimilar(self, grid1, grid2):
        # Measure how similar two grids are.
        totalActiveIns = np.sum(self.grid1 != 0)
        totalAndGrids = np.sum(np.logical_and(grid1 != 0, self.grid2 != 0))
        percentSimilar = float(totalAndGrids) / float(totalActiveIns)
        return percentSimilar

    def test_case1(self):
        '''
        '''
        self.nSteps(100)

        app = QtGui.QApplication(sys.argv)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        sys.exit(app.exec_())

        assert 1 == 1



