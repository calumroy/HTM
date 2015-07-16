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

testParameters = {
                    'HTM':
                        {
                        'numLevels': 1,
                        'columnArrayWidth': 80,
                        'columnArrayHeight': 20,
                        'cellsPerColumn': 10,

                        'HTMRegions': [{
                            'numLayers': 1,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 2,
                                'centerPotSynapses': 1,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 5,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'dutyCycleAverageLength': 1000,
                                'synPermanence': 0.4,

                                'Columns': [{
                                    'minOverlap': 3,
                                    'boost': 1,
                                    'inhibitionRadius': 2,
                                    'potentialWidth': 10,
                                    'potentialHeight': 10,
                                    'spatialPermanenceInc': 0.1,
                                    'spatialPermanenceDec': 0.02,
                                    'permanenceInc': 0.1,
                                    'permanenceDec': 0.02,
                                    'minDutyCycle': 0.01,
                                    'boostStep': 0,
                                    'historyLength': 2
                                }]
                            }]
                        }]
                    }
                }


class test_RunTime:
    def setUp(self):
        '''
        We are tesing the run time of the HTM network.

        A set number of input sequeces are fed into the htm.

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
        Run time testing.

        Run the HTM for a certain number of steps so synapses are created.
        Then run one step and see the resulting profile

        '''
        # Let the spatial pooler learn spatial patterns.
        self.nSteps(1)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.nSteps(1)

        from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        # app = QtGui.QApplication(sys.argv)
        # self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        # sys.exit(app.exec_())

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        assert 1 == 1
