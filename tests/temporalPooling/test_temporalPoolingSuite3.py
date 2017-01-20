from HTM_network import HTM_network
import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
from copy import deepcopy
from utilities import imageInputs as imageIn, measureTemporalPooling as mtp
from utilities import sdrFunctions
from utilities import startHtmGui as htmgui


testParameters = {
                  'HTM': {
                        'numLevels': 1,
                        'columnArrayWidth': 10,
                        'columnArrayHeight': 20,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 1,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 1,
                                'minOverlap': 8,
                                'inhibitionWidth': 12,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'potentialWidth': 28,
                                'potentialHeight': 8,
                                'spatialPermanenceInc': 0.16,
                                'spatialPermanenceDec': 0.02,
                                'activeColPermanenceDec': 0.001,
                                'tempDelayLength': 3,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
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


class test_temporalPoolingSuite3:
    def setUp(self):
        '''

        This test will use multiple regions in one level.
        This is because the regions ouputs simply pass onto the next,
        their is no compilcated feedback happening with one level.

        The test parameters are loaded from a jason file.

        To test the temporal pooling ability of the regions a sequence
        of inputs are repeatably inputted to the HTM. If after a number
        of steps the top most layer is only changing slightly compared to
        the bottom layer then temporal pooling is occuring.
        '''

        params = testParameters

        # Create an array of inputs which will be fed to the htm so it
        # can try to temporarily pool them.

        #inputWidth = params['HTM']['columnArrayWidth']
        #inputHeight = params['HTM']['columnArrayHeight']

        self.InputCreator = imageIn.imageInputs(r'test_seqs')
        #self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)
        self.htm = HTM_network.HTM(self.InputCreator.createSimGrid(),
                                   params
                                   )

        # Measure the temporal pooling
        self.temporalPooling = mtp.measureTemporalPooling()

        # define the gui class
        self.gui = htmgui.start_htm_gui()

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

    def test_tempEquality(self):
        '''
        Test to make sure that a pattern is eventually temporally
        pooled by cells that where active at different times of the
        input pattern initially.

        Eg. pattern A, B, C should be pooled into a stable pattern
        with some cells that originally became active for input A and some that
        where originally active for B and C as well.
        '''
        self.InputCreator.changePattern(0)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.nSteps(100)

        self.InputCreator.changePattern(1)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.nSteps(101)
