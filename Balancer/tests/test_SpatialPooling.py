from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli


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

        self.InputCreator = svli.simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)
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
        #self.nSteps(100)

        app = QtGui.QApplication(sys.argv)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        sys.exit(app.exec_())

        assert 1 == 1



