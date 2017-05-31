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
                        'columnArrayWidth': 11,
                        'columnArrayHeight': 31,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 3,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 1,
                                'minOverlap': 3,
                                'wrapInput':0,
                                'inhibitionWidth': 4,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'potentialWidth': 5,
                                'potentialHeight': 5,
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
                                'wrapInput':0,
                                'inhibitionWidth': 8,
                                'inhibitionHeight': 4,
                                'centerPotSynapses': 1,
                                'potentialWidth': 7,
                                'potentialHeight': 7,
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
                                'wrapInput':1,
                                'inhibitionWidth': 30,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'connectPermanence': 0.3,
                                'potentialWidth': 34,
                                'potentialHeight': 31,
                                'spatialPermanenceInc': 0.1,
                                'spatialPermanenceDec': 0.01,
                                'activeColPermanenceDec': 0.01,
                                'tempDelayLength': 10,
                                'permanenceInc': 0.15,
                                'permanenceDec': 0.05,
                                'tempSpatialPermanenceInc': 0.2,
                                'tempSeqPermanenceInc': 0.1,
                                'minThreshold': 5,
                                'minScoreThreshold': 3,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'colSynPermanence': 0.0,
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

        # Create an array of inputs which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputHeight = 2*params['HTM']['columnArrayHeight']
        #import ipdb; ipdb.set_trace()
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

    def getLearningCellsOutput(self, htm, level, layer):
        # From the level for the given htm network
        # get the cells which are in the laerning state in a 2d array form.
        # The grid should contain only ones and zeros corresponding to
        # a cells location. One if that cell is learning zero otherwise.
        learnCellsGrid = htm.regionArray[level].layerArray[layer].getLearnCellsOutput()

        return learnCellsGrid

    def test_tempEquality(self):
        '''
        Test to make sure that a pattern is eventually temporally
        pooled by cells that where active at different times of the
        input pattern initially.

        Eg. pattern A, B, C should be pooled into a stable pattern
        with some cells that originally became active for input A and some that
        where originally active for B and C as well.

        We do this for an input patterns of moving vertical lines.
        Two input patterns are used to make sure both input patterns pool over their own
        input sequences and do not overlap with each other very much.
        '''
        # How many patterns are we comparing against
        numPatternsTested = 2
        # The level and layer in the htm we are testing on.
        level = 0
        layer = 2

        # Select which patterns are to be tested against. See the simpleVertical lines calcualtor.
        # These two patterns are similar but offset. None of the inputs overlap between the patterns.
        pattern1_ind = 11
        pattern2_ind = 12

        pat1NumInputs = self.InputCreator.getNumInputsPat(pattern1_ind)
        pat2NumInputs = self.InputCreator.getNumInputsPat(pattern2_ind)
        # This test will only work if both patterns have the same number of inputs.
        assert pat1NumInputs == pat2NumInputs
        numInputs = pat1NumInputs

        # self.InputCreator.changePattern(pattern1_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

        # self.InputCreator.changePattern(13)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

        # Create arrays to store the output SDR from the HTM layer for each input.
        outputSDR00 = self.getLearningCellsOutput(self.htm, level, layer)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(numInputs)]
                                             for p in range(numPatternsTested)]
                                            )

        # Set the input to the first pattern
        self.InputCreator.changePattern(pattern1_ind)
        # Since the higher layers take longer to receive a new input run through the
        # test pattern once. So the higher layer has at least got an intial input from the pattern.
        self.nSteps(numInputs)
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        # Now run through the pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        for i in range(numInputs):
            outputsFromPatternX[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        #import ipdb; ipdb.set_trace()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        # Since the higher layers take longer to receive a new input run through the
        # test pattern once. So the higher layer has at least got an intial input from the new pattern.
        self.nSteps(2*numInputs)
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        # Store the outputs from the second pattern.
        for i in range(numInputs):
            outputsFromPatternX[1][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Set the input to the first pattern
        self.InputCreator.changePattern(pattern1_ind)
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs multiple times and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(40*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        # Rerun through all the inputs multiple times and store the outputs from
        # the second pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(40*numInputs)
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
            print "simularity of outputs in pattern 1 sequence[%s] = %s" % (i, simOutIn1[i])
            print "simularity of outputs in pattern 2 sequence[%s] = %s" % (i, simOutIn2[i])
            print "simularity of outputs in pattern 1 vs 2 initial learning[%s] = %s" % (i, simOut1vs2start[i])
            print "simularity of outputs in pattern 1 vs 2 final patterns[%s] = %s" % (i, simOut1vs2end[i])

            assert simOut1vs2start[i] < 0.25
            assert simOut1vs2end[i] < 0.25

        # self.InputCreator.changePattern(pattern1_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.InputCreator.setIndex(0)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

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
        layer = 2

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
        self.nSteps(2*numInputs)
        for i in range(numInputs):
            outputsFromPatternXAgain[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        #self.gui.startHtmGui(self.htm, self.InputCreator)
        # Rerun through all the inputs multiple times and store the outputs from
        # the second pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(2*numInputs)
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
            assert (simOut1vs2end[i] < 0.1)

        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)

    def test_tempDiffPooled(self):
        '''
        Test to make sure that when two patterns are different then the
        temporally pooled outputs from a layer for each sequence
        are initially different too. After enough transitions from one pattern to the other
        this transition should also be temporally pooled resulting in a single
        stable top layer output. 

        We do this for two input patterns of vertical lines.
        '''

        # How many patterns are we comparing against
        numPatternsTested = 2
        # The level and layer in the htm we are testing on.
        level = 0
        layer = 2

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

        # Creat a tensor to store the output of each pattern
        outputSDR00 = self.getLearningCellsOutput(self.htm, level, layer)
        o_patX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        o_patXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(numInputs)]
                                             for p in range(numPatternsTested)]
                                            )
        # Now run through the pattern and store each output SDR
        # These are used to compare against later on.
        # o_patX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        self.nSteps(40*numInputs)
        for i in range(numInputs):
            o_patX[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        #import ipdb; ipdb.set_trace()

        # Set the input to the second pattern
        self.InputCreator.changePattern(pattern2_ind)
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(40*numInputs)
        for i in range(numInputs):
            o_patX[1][i] = self.getLearningCellsOutput(self.htm, level, layer)
            self.step()

        # Now the temporally pooled outputs of both patterns have been stored.
        # We want to make sure that if the transition from on pattern to the next occurs enough then
        # this is also temporally pooled. It should also result in a temporally pooled pattern consisting of
        # half of each of the output form the individual temporal pooled patterns.
        for i in range(20*numInputs):
            self.InputCreator.changePattern(pattern1_ind)
            self.nSteps(numInputs)
            self.InputCreator.changePattern(pattern2_ind)
            self.nSteps(numInputs)

        
        # Rerun through all the inputs and store the final temporally pooled pattern
        # Also measure the amount of temporal pooling occuring.
        # Measure the temporal pooling
        tempMeasure1 = mtp.measureTemporalPooling()
        tempMeasure2 = mtp.measureTemporalPooling()
        tempMeasure12 = mtp.measureTemporalPooling()
        tempPoolPercent1 = 0.0
        tempPoolPercent2 = 0.0
        tempPoolPercent12 = 0.0
        self.InputCreator.changePattern(pattern1_ind)
        self.InputCreator.setIndex(0)
        for i in range(numInputs):
            o_patXAgain[0][i] = self.getLearningCellsOutput(self.htm, level, layer)
            tempPoolPercent1 = tempMeasure1.temporalPoolingPercent(o_patXAgain[0][i])
            tempPoolPercent12 = tempMeasure12.temporalPoolingPercent(o_patXAgain[0][i])
            self.step()
        self.InputCreator.changePattern(pattern2_ind)
        self.InputCreator.setIndex(0)
        for i in range(numInputs):
            o_patXAgain[1][i] = self.getLearningCellsOutput(self.htm, level, layer)
            tempPoolPercent2 = tempMeasure2.temporalPoolingPercent(o_patXAgain[1][i])
            tempPoolPercent12 = tempMeasure12.temporalPoolingPercent(o_patXAgain[1][i])
            self.step()

        # Now we need to compare the two outputs from both times the two input patterns
        # were stored.
        simOut1Vs2 = np.zeros(numInputs)
        simOut1vs2end = np.zeros(numInputs)
        simOut1vsEnd= np.zeros(numInputs)
        simOut2vsEnd= np.zeros(numInputs)
        
        for i in range(numInputs):
            
            simOut1Vs2[i] = sdrFunctions.similarInputGrids(o_patX[0][i],
                                                           o_patX[1][i])
            simOut1vs2end[i] = sdrFunctions.similarInputGrids(o_patXAgain[0][i],
                                                              o_patXAgain[1][i])
            simOut1vsEnd[i] = sdrFunctions.similarInputGrids(o_patX[0][i],
                                                              o_patXAgain[0][i])
            simOut2vsEnd[i] = sdrFunctions.similarInputGrids(o_patX[1][i],
                                                              o_patXAgain[1][i])
            # # The simularity between the inital HTM output for each pattern vs the final output.
            print "simularity of temporally pooled pattern 1 vs pattern 2 sequence[%s] = %s" % (i, simOut1Vs2[i])
            print "simularity of outputs in pattern 1 vs 2 after final temp pooled patterns[%s] = %s" % (i, simOut1vs2end[i])
            print "simularity of outputs in pattern 1 vs final temp pooled patterns[%s] = %s" % (i, simOut1vsEnd[i])
            print "simularity of outputs in pattern 2 vs final temp pooled patterns[%s] = %s" % (i, simOut2vsEnd[i])
            assert (simOut1Vs2[i] < 0.1)
            assert (simOut1vs2end[i] > 0.9)
            assert (simOut1vsEnd[i] > 0.0)
            assert (simOut2vsEnd[i] > 0.0)

        print "The percentage of temporal pooling occuring in the:\n"
        print "      pattern 1 inputs = %s" %tempPoolPercent1
        print "      pattern 2 inputs = %s" %tempPoolPercent2
        print "      final total temporally pooled 1 and 2 patterns = %s" %tempPoolPercent12 
        assert (tempPoolPercent1 > 0.9)
        assert (tempPoolPercent2 > 0.9)
        assert (tempPoolPercent12 > 0.9)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern1_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)
        # self.InputCreator.changePattern(pattern2_ind)
        # self.gui.startHtmGui(self.htm, self.InputCreator)




