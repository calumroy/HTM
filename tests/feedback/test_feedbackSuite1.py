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
                            'numLayers': 2,
                            'enableHigherLevFb': 1,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 1,
                                'minOverlap': 3,
                                'wrapInput':1,
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
                                'minOverlap': 4,
                                'wrapInput':1,
                                'inhibitionWidth': 8,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'potentialWidth': 20,
                                'potentialHeight': 10,
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
                                }]
                            }]
                        }
                    }


class test_feedbackSuite1:
    def setUp(self):
        '''

        This test will use multiple regions in one level.
        This is because the regions outputs simply pass onto the next,
        their is no compilcated feedback happening with one level.

        The test parameters are loaded from a jason file.

        '''

        params = testParameters

        self.InputCreator = imageIn.imageInputs(r'test_seqs_suite1')
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

    def getLearningCellsOutput(self, htm, level, layer):
        # From the level for the given htm network
        # get the cells which are in the laerning state in a 2d array form.
        # The grid should contain only ones and zeros corresponding to
        # a cells location. One if that cell is learning zero otherwise.
        learnCellsGrid = htm.regionArray[level].layerArray[layer].getLearnCellsOutput()

        return learnCellsGrid


    def test_temporalDiff(self):
        '''
        Test to make sure that when two patterns are different then the
        temporally pooled outputs from a layer for each sequence
        are different too.

        We do this for an input patterns of vertical lines.
        '''

        # How many patterns are we comparing against
        numPatternsTested = 2
        # The level and layer in the htm we are testing on.
        level = 0
        layer = 1

        pattern1_ind = 8
        pattern2_ind = 9

        pat1NumInputs = self.InputCreator.getNumInputsPat(pattern1_ind)
        pat2NumInputs = self.InputCreator.getNumInputsPat(pattern2_ind)
        # This test will only work if both patterns have the same number of inputs.
        assert pat1NumInputs == pat2NumInputs
        numInputs = pat1NumInputs

        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

        # Set the input to the first pattern
        self.InputCreator.changePattern(pattern1_ind)

        # Now run through the pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        self.nSteps(20*numInputs)
        outputSDR00 = self.getLearningCellsOutput(self.htm, level, layer)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(numInputs)]
                                             for p in range(numPatternsTested)]
                                            )
        for i in range(numInputs):
            outputsFromPatternX[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        #import ipdb; ipdb.set_trace()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(20*numInputs)
        for i in range(numInputs):
            outputsFromPatternX[1][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Set the input to the first pattern
        self.InputCreator.changePattern(pattern1_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs multiple times and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(1*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        # Rerun through all the inputs multiple times and store the outputs from
        # the second pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(1*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[1][i] = self.getLearningCellsOutput(self.htm, level, layer)
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

            # # The simularity between the inital HTM output for each pattern vs the final output.
            # print "simularity of outputs in pattern 1 sequence[%s] = %s" % (i, simOutIn1[i])
            # print "simularity of outputs in pattern 2 sequence[%s] = %s" % (i, simOutIn2[i])
            # print "simularity of outputs in pattern 1 vs 2 initial learing[%s] = %s" % (i, simOut1vs2start[i])
            print "simularity of outputs in pattern 1 vs 2 final patterns[%s] = %s" % (i, simOut1vs2end[i])

            # The simularity between the outputs from the 2 patterns for each input should be quite small
            # since the patterns did not share any input features.
            assert (simOut1vs2end[i] < 0.15)

        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)




