from HTM_network import HTM_network
import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli, measureTemporalPooling as mtp
from utilities import sdrFunctions
from utilities import startHtmGui as htmgui


testParameters = {
                  'HTM': {
                        'numLevels': 1,
                        'columnArrayWidth': 10,
                        'columnArrayHeight': 30,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 3,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 1,
                                'minOverlap': 3,
                                'inhibitionWidth': 4,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'potentialWidth': 4,
                                'potentialHeight': 4,
                                'spatialPermanenceInc': 0.1,
                                'spatialPermanenceDec': 0.02,
                                'activeColPermanenceDec': 0.02,
                                'tempDelayLength': 3,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
                                'tempSpatialPermanenceInc': 0,
                                'tempSeqPermanenceInc': 0,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 5,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'colSynPermanence': 0.1,
                                'cellSynPermanence': 0.4
                                },
                                {
                                'desiredLocalActivity': 1,
                                'minOverlap': 2,
                                'inhibitionWidth': 8,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'potentialWidth': 8,
                                'potentialHeight': 8,
                                'spatialPermanenceInc': 0.2,
                                'spatialPermanenceDec': 0.02,
                                'activeColPermanenceDec': 0.02,
                                'tempDelayLength': 3,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
                                'tempSpatialPermanenceInc': 0.2,
                                'tempSeqPermanenceInc': 0.1,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 3,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'colSynPermanence': 0.1,
                                'cellSynPermanence': 0.4
                                },
                                {
                                'desiredLocalActivity': 1,
                                'minOverlap': 2,
                                'inhibitionWidth': 15,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'connectPermanence': 0.3,
                                'potentialWidth': 20,
                                'potentialHeight': 10,
                                'spatialPermanenceInc': 0.2,
                                'spatialPermanenceDec': 0.02,
                                'activeColPermanenceDec': 0.02,
                                'tempDelayLength': 10,
                                'permanenceInc': 0.15,
                                'permanenceDec': 0.05,
                                'tempSpatialPermanenceInc': 0.2,
                                'tempSeqPermanenceInc': 0.15,
                                'minThreshold': 5,
                                'minScoreThreshold': 3,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'colSynPermanence': 0.1,
                                'cellSynPermanence': 0.4
                                }]
                            }]
                        }
                    }


class test_temporalPoolingSuite4:
    def setUp(self):
        '''

        This test will use multiple regions in one level.
        This is because the regions outputs simply pass onto the next,
        their is no compilcated feedback happening with one level.

        The test parameters are loaded from a jason file.

        To test the temporal pooling ability of the regions a sequence
        of inputs are repeatably inputted to the HTM. If after a number
        of steps the top most layer is only changing slightly compared to
        the bottom layer then temporal pooling is occuring.
        '''

        params = testParameters

        self.width = 10
        self.height = 30
        self.cellsPerColumn = 3
        self.numLevels = 1
        self.numLayers = 3

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputHeight = 2*params['HTM']['columnArrayHeight']

        self.InputCreator = svli.simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)
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
        # Update the HTM input and run through the HTM.
        newInput = self.InputCreator.createSimGrid()
        self.htm.spatialTemporal(newInput)
        if (self.htm.regionArray[0].layerArray[0].timeStep % 20 == 0):
            print " TimeStep = %s" % self.htm.regionArray[0].layerArray[0].timeStep

    def nSteps(self, numSteps):
        print "Running HTM for %s steps" % numSteps
        for i in range(numSteps):
            self.step()

    def getCellsGridOutput(self, htm, level, layer):
        # From the level for the given htm network
        # get the layers output in a 2d array form.
        # The grid should contain only ones and zeros corresponding to
        # a cells location. One if that cell is active zero otherwise.
        activeColGrid = htm.regionArray[level].layerOutput(layer)

        return activeColGrid

    def test_temporalDiff(self):
        '''
        Test to make sure that when two patterns are similar then the
        temporally pooled outputs from a layer for each sequence
        are similar too.

        We do this for an input patterns of vertical lines.
        '''
        
        # How many patterns are we comparing against
        numPatternsTested = 2
        # The level and layer in the htm we are testing on.
        level = 0
        layer = 2

        pattern1_ind = 6
        pattern2_ind = 7

        # self.InputCreator.changePattern(2)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(1)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

        # Set the input to the first pattern 
        self.InputCreator.changePattern(pattern1_ind)
        numInputs = self.InputCreator.numInputs
        
        # Now run through the pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        self.nSteps(2*numInputs)
        outputSDR00 = self.getCellsGridOutput(self.htm, level, layer)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(numInputs)]
                                             for p in range(numPatternsTested)]
                                            )
        for i in range(numInputs):
            outputsFromPatternX[0][i] = self.getCellsGridOutput(self.htm, level, layer)
            self.step()

        #import ipdb; ipdb.set_trace()

        # Set the input to the second pattern 
        self.InputCreator.changePattern(pattern2_ind)
        numInputs = self.InputCreator.numInputs
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(2*numInputs)
        for i in range(numInputs):
            outputsFromPatternX[1][i] = self.getCellsGridOutput(self.htm, level, layer)
            self.step()

        # Set the input to the first pattern     
        self.InputCreator.changePattern(pattern1_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        numInputs = self.InputCreator.numInputs
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs multiple times and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(20*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[0][i] = self.getCellsGridOutput(self.htm, level, layer)
            self.step()

        # Set the input to the second pattern 
        self.InputCreator.changePattern(pattern2_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        numInputs = self.InputCreator.numInputs
        # Rerun through all the inputs multiple times and store the outputs from
        # the second pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(20*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[1][i] = self.getCellsGridOutput(self.htm, level, layer)
            self.step()

        # Now we need to compare the two outputs from both times the two input patterns
        # were stored.
        simOutIn1 = np.zeros(numInputs)
        simOutIn2 = np.zeros(numInputs)
        simOut1vs2start = np.zeros(numInputs)
        simOut1vs2end = np.zeros(numInputs)
        for i in range(numInputs):
            simOutIn1[i] = sdrFunctions.similarInputGrids(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            simOutIn2[i] = sdrFunctions.similarInputGrids(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])
            simOut1vs2start[i] = sdrFunctions.similarInputGrids(outputsFromPatternX[0][i],
                                                     outputsFromPatternX[1][i])
            simOut1vs2end[i] = sdrFunctions.similarInputGrids(outputsFromPatternXAgain[0][i],
                                                     outputsFromPatternXAgain[1][i])

            # The simularity between the inital HTM output for each pattern vs the final output.
            print "simularity of outputs in pattern 1 sequence[%s] = %s" % (i, simOutIn1[i])
            #assert simOutIn1[i] > 0.2
            print "simularity of outputs in pattern 2 sequence[%s] = %s" % (i, simOutIn2[i])
            #assert simOutIn2[i] > 0.2
            print "simularity of outputs in pattern 1 vs 2 initial learning[%s] = %s" % (i, simOut1vs2start[i])
            #assert simOut1vs2start[i] < 0.25
            print "simularity of outputs in pattern 1 vs 2 learnt patterns[%s] = %s" % (i, simOut1vs2end[i])
            #assert simOut1vs2end[i] < 0.25

        # Now the inital learning for patterns 1 and 2 would have produced bursting.
        # The learnt temporally pooled patterns for patterns 1 and 2 should be around 33% similar
       
        #assert (simOut1vs2start[2] > 0.9)
        #assert (simOut1vs2end[0] > 0.2) and (simOut1vs2end[0] < 0.5)
        #assert (simOut1vs2end[1] > 0.2) and (simOut1vs2end[1] < 0.5)
        #assert (simOut1vs2end[2] > 0.2) and (simOut1vs2end[2] < 0.5)
        
        self.InputCreator.changePattern(2)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.InputCreator.changePattern(1)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.InputCreator.changePattern(2)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.InputCreator.changePattern(1)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.InputCreator.changePattern(3)
        self.gui.startHtmGui(self.htm, self.InputCreator)
        self.InputCreator.changePattern(1)
        self.gui.startHtmGui(self.htm, self.InputCreator)



