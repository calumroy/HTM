from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli
from utilities import sdrFunctions as SDRFunct

testParameters = {
                    'HTM':
                        {
                        'numLevels': 1,
                        'columnArrayWidth': 10,
                        'columnArrayHeight': 30,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 1,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 2,
                                'minOverlap': 3,
                                'inhibitionWidth': 3,
                                'inhibitionHeight': 3,
                                'centerPotSynapses': 1,
                                'potentialWidth': 4,
                                'potentialHeight': 4,
                                'spatialPermanenceInc': 0.1,
                                'spatialPermanenceDec': 0.02,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 5,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'dutyCycleAverageLength': 1000,
                                'colSynPermanence': 0.2,
                                'cellSynPermanence': 0.4,

                                'Columns': [{
                                    'boost': 1,
                                    'minDutyCycle': 0.01,
                                    'boostStep': 0,
                                    'historyLength': 2
                                }]
                            }]
                        }]
                    }
                }


class test_spatialPooling:
    def setUp(self):
        '''
        We are tesing the spatial pooler.

        A set number of input sequeces are fed into the htm.
        The spatial pooler over time selects certian columns to
        represent certain features of the input. The spatial pooler
        should allow similar inputs to activate similar columns since they contain
        similar features.

        '''

            # Open and import the parameters .json file
        #with open('testSpatialPooling.json', 'r') as paramsFile:
        params = testParameters

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputHeight = 2*params['HTM']['columnArrayHeight']

        self.InputCreator = svli.simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)

        self.htm = HTM(self.InputCreator.createSimGrid(),
                       params
                       )

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

        SDR1 = self.InputCreator.inputs[0][0]
        SDR2 = self.InputCreator.inputs[0][self.InputCreator.numInputs-1]

        combinedInput = SDRFunct.orSDRPatterns(SDR1, SDR2)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.htm.spatialTemporal(SDR1)
        colSDR1 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(SDR2)
        colSDR2 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(combinedInput)
        combinedOutput = self.getColumnGridOutput(self.htm, 0, 0)

        similarPerIn1 = self.gridsSimilar(colSDR1, combinedOutput)
        similarPerIn2 = self.gridsSimilar(colSDR2, combinedOutput)

        # app = QtGui.QApplication(sys.argv)
        # self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        # sys.exit(app.exec_())

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        assert similarPerIn1 >= 0.49 and similarPerIn1 <= 0.51
        assert similarPerIn2 >= 0.49 and similarPerIn2 <= 0.51

    def test_case2(self):
        '''
        The same as test 1 expect use input patterns that are much closer to each other.

        Make sure the patterns are far enough apart that the active columns
        can't inhibit one another. If this happens then the combined pattern will be
        registered as a totally new pattern and new columns will win inhibition.
        This type of behaviour is expected but unwanted for this test.
        '''

        # Let the spatial pooler learn spatial patterns.
        self.nSteps(150)

        numInps = self.InputCreator.numInputs
        middleInp = int(numInps/2)
        # Note 6 was used to seperate the inputs by enough so an inhibition
        # radius of one with the current input size doesn't cause the old columns to
        # inhibit one another for the new combined input.
        middleInpPlus = middleInp + 6
        # Make sure this index is still smaller then numInps
        if middleInpPlus >= numInps:
            middleInpPlus = 0

        SDR1 = self.InputCreator.inputs[0][middleInp]
        SDR2 = self.InputCreator.inputs[0][middleInpPlus]

        combinedInput = SDRFunct.orSDRPatterns(SDR1, SDR2)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.htm.spatialTemporal(SDR1)
        colSDR1 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(SDR2)
        colSDR2 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(combinedInput)
        combinedOutput = self.getColumnGridOutput(self.htm, 0, 0)

        #app = QtGui.QApplication(sys.argv)
        #self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        #sys.exit(app.exec_())

        # Compare the combined output SDR to the individual input SDRs.
        # The combined one should be equal to 50% of the individual ones.
        similarPerIn1 = self.gridsSimilar(colSDR1, combinedOutput)
        similarPerIn2 = self.gridsSimilar(colSDR2, combinedOutput)

        assert similarPerIn1 >= 0.49 and similarPerIn1 <= 0.51
        assert similarPerIn2 >= 0.49 and similarPerIn2 <= 0.51

    def test_case3(self):
        '''
        The same as test 1 expect use half of the two chosen patterns.

        Make sure the patterns are far enough apart that the active columns
        can't inhibit one another. If this happens then the combined pattern will be
        registered as a totally new pattern and new columns will win inhibition.
        This type of behaviour is expected but unwanted for this test.
        '''

        # Let the spatial pooler learn spatial patterns.
        self.nSteps(150)

        SDR1 = self.InputCreator.inputs[0][0]
        SDR2 = self.InputCreator.inputs[0][self.InputCreator.numInputs-1]

        # Choose a half of the active columns from each SDR to keep on.
        totalActiveIns1 = np.sum(SDR1 != 0)
        totalActiveIns2 = np.sum(SDR2 != 0)
        numTurnedOff1 = 0
        numLeftOn2 = 0
        for y in range(len(SDR1)):
            for x in range(len(SDR1[0])):
                if SDR1[y][x] == 1 and numTurnedOff1 < round(totalActiveIns1/2):
                    SDR1[y][x] = 0
                    numTurnedOff1 += 1

        for y in range(len(SDR2)):
            for x in range(len(SDR2[0])):
                if SDR2[y][x] == 1:
                    if numLeftOn2 > round(totalActiveIns2/2):
                        SDR2[y][x] = 0
                    numLeftOn2 += 1

        combinedInput = SDRFunct.orSDRPatterns(SDR1, SDR2)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.htm.spatialTemporal(SDR1)
        colSDR1 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(SDR2)
        colSDR2 = self.getColumnGridOutput(self.htm, 0, 0)
        self.htm.spatialTemporal(combinedInput)
        combinedOutput = self.getColumnGridOutput(self.htm, 0, 0)

        similarPerIn1 = self.gridsSimilar(colSDR1, combinedOutput)
        similarPerIn2 = self.gridsSimilar(colSDR2, combinedOutput)

        # app = QtGui.QApplication(sys.argv)
        # self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        # sys.exit(app.exec_())

        # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        # Not exactly 50% since on the edges of the two patterns new columns may activate.
        assert similarPerIn1 >= 0.44 and similarPerIn1 <= 0.56
        assert similarPerIn2 >= 0.44 and similarPerIn2 <= 0.56
