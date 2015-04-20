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
        if (self.htm.regionArray[0].layerArray[0].timeStep % 20 == 0):
            print " TimeStep = %s" % self.htm.regionArray[0].layerArray[0].timeStep

    def nSteps(self, numSteps):
        print "Running HTM for %s steps" % numSteps
        for i in range(numSteps):
            self.step()

    def gridsSimilar(self, grid1, grid2):
        # Measure how similar two grids are.
        totalActiveIns1 = np.sum(grid1 != 0)
        totalActiveIns2 = np.sum(grid2 != 0)
        totalAndGrids = np.sum(np.logical_and(grid1 != 0, grid2 != 0))
        totalActiveIns = totalActiveIns1 + totalActiveIns2 - totalAndGrids
        percentSimilar = float(totalAndGrids) / float(totalActiveIns)
        return percentSimilar

    def getColumnGridOutput(self, htm, level, layer):
        # From the level for the given htm network
        # get the active columns in a 2d array form.
        # The grid should contain only ones and zeros corresponding to
        # a columns location. One if that column is active zero otherwise.
        activeCols = htm.regionArray[level].layerArray[layer].activeColumns
        width = htm.regionArray[level].layerArray[layer].width
        height = htm.regionArray[level].layerArray[layer].height
        activeColGrid = np.array([[0 for i in range(width)] for j in range(height)])

        for column in activeCols:
            activeColGrid[column.pos_y][column.pos_x] = 1

        return activeColGrid

    def test_case1(self):
        '''
        Spatial pooler superposition testing.

        Test the spatial pooler and make sure that an input
        that contains features from two different input patterns
        creates an output where the same columns that are activated
        for both the inputs are still activated for the combined input.

        Note the output SDR may be different as the new combine input
        will not be in sequence. The same columns should be active just
        not the same cells.

        '''
        # Let the spatial pooler learn spatial patterns.
        self.nSteps(150)

        SDR1 = self.InputCreator.inputs[0]
        SDR2 = self.InputCreator.inputs[self.InputCreator.numInputs-1]

        combinedInput = self.InputCreator.orSDRPatterns(SDR1, SDR2)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.htm.spatialTemporal(SDR1)
        colSDR1 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(SDR2)
        colSDR2 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(combinedInput)
        combinedOutput = self.getColumnGridOutput(self.htm, 0, 0)

        similarPerIn1 = self.gridsSimilar(colSDR1, combinedOutput)
        similarPerIn2 = self.gridsSimilar(colSDR2, combinedOutput)

        #app = QtGui.QApplication(sys.argv)
        #self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        #sys.exit(app.exec_())

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        assert similarPerIn1 >= 0.49 and similarPerIn1 <= 0.51
        assert similarPerIn2 >= 0.49 and similarPerIn2 <= 0.51
