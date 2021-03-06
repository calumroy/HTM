import cProfile
from HTM_network import HTM_network
import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli
from utilities import startHtmGui as htmgui

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

testParameters = {
                    'HTM':
                        {
                        'numLevels': 1,
                        'columnArrayWidth': 60,
                        'columnArrayHeight': 60,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 1,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 2,
                                'minOverlap': 2,
                                'wrapInput':1,
                                'inhibitionWidth': 2,
                                'inhibitionHeight': 3,
                                'centerPotSynapses': 1,
                                'potentialWidth': 34,
                                'potentialHeight': 31,
                                'spatialPermanenceInc': 0.1,
                                'spatialPermanenceDec': 0.02,
                                'activeColPermanenceDec': 0.02,
                                'tempDelayLength': 3,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
                                'tempSpatialPermanenceInc': 0.2,
                                'tempSeqPermanenceInc': 0.1,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 5,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'colSynPermanence': 0.1,
                                'cellSynPermanence': 0.4
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
        # with open('testSpatialPooling.json', 'r') as paramsFile:
        params = testParameters

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputHeight = 2*params['HTM']['columnArrayHeight']

        self.InputCreator = svli.simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)

        self.htm = HTM_network.HTM(self.InputCreator.createSimGrid(), params)

        # Setup some parameters of the HTM
        self.setupParameters()

        # define the gui class
        self.gui = htmgui.start_htm_gui()

    def setupParameters(self):
        # Setup some parameters of the HTM
        pass

    @do_cprofile  # For profiling
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

    def test_case1(self):
        '''
        Run time testing.

        Run the HTM for a certain number of steps so synapses are created.
        Then run one step and see the resulting profile

        '''
        # Let the spatial pooler learn spatial patterns.
        self.nSteps(1)

        # Run the inputs through the htm just once and obtain the column SDR outputs.
        self.nSteps(5)

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        self.gui.startHtmGui(self.htm, self.InputCreator)

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        assert 1 == 1

