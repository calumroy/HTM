
from HTM_Balancer import HTM
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from utilities import customSDRInputs as seqInputs
from utilities import startHtmGui as gui
import math

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
                                'desiredLocalActivity': 1,
                                'minOverlap': 2,
                                'inhibitionWidth': 3,
                                'inhibitionHeight': 4,
                                'centerPotSynapses': 1,
                                'potentialWidth': 4,
                                'potentialHeight': 4,
                                'spatialPermanenceInc': 0.1,
                                'spatialPermanenceDec': 0.02,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
                                'connectPermanence': 0.3,
                                'minThreshold': 5,
                                'minScoreThreshold': 3,
                                'newSynapseCount': 10,
                                'maxNumSegments': 10,
                                'activationThreshold': 6,
                                'dutyCycleAverageLength': 1000,
                                'synPermanence': 0.4,

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


class test_spatialPoolingSuite3:
    def setUp(self):
        '''
        We are testing the spatial pooler, and it's
        ability to  work with the temporal pooler to remember multiple sequences.

        '''

        params = testParameters

        numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        self.inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        self.inputHeight = 2*params['HTM']['columnArrayHeight']

        self.InputCreator = seqInputs.customSDRInputs(self.inputWidth, self.inputHeight, numInputs)

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
        # Return a percentage describing how similar the two
        # input binary grids are.
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

    def createDottedVerticalLineSeq(self, patternLength, lineWidth, dottedLineHeight=-1, dottedLineGap=1):
        # Create the input patterns to test with.
        patNumInputs = patternLength
        newPattern = np.array([[[0 for i in range(self.inputWidth)]
                                for j in range(self.inputHeight)]
                              for k in range(patNumInputs)])
        # The first pattern is just a dotted vertical line of width 1.
        # It moves left to right.
        for n in range(len(newPattern)):
            for y in range(len(newPattern[0])):
                for x in range(len(newPattern[n][y])):
                    if (x >= n-int(math.floor(lineWidth/2.0))) and (x <= n+int(math.ceil(lineWidth/2.0))-1):
                        if dottedLineHeight > 0:
                            if (y % (dottedLineHeight+1)) < dottedLineGap:
                                newPattern[n][y][x] = 1
                        else:
                            # No dotted line
                            newPattern[n][y][x] = 1
        # Store this pattern.
        self.InputCreator.appendSequence(newPattern)

    def test_case1(self):
        '''
        Spatial pooler multiple input patterns.

        Test the spatial pooler and make sure that if it learns
        sequence A then another sequence B, A should not be forgotten.

        Sequence B is very similar to sequence A it contains extra
        on bits compared to A. Sequence A is a vertical line moving left to right.
        Sequence B is a slightly thicker vertical line moving left to right.

        Also make sure that the output from the htm for the two input sequeces is
        similar as both patterns share the same features.

        '''
        # Create the input patterns to test with.
        # Create a vertical line that moves left to right with a
        # line width of 2.
        lineWidth = 2
        patNumInputs = self.InputCreator.getNumInputsInSeq(0)
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth)

        # We will use these defined pattern above for testing.
        # We are also using the default pattern (a vertical line moving left ot right).
        numPatternsTested = self.InputCreator.getNumCustomSequences()+1
        pat0NumInputs = self.InputCreator.getNumInputsInSeq(0)
        pat1NumInputs = self.InputCreator.getNumInputsInSeq(1)

        # Run the patterns through the htm multiple times
        # this is done so the htm can settle on a representation
        # for each input.
        self.InputCreator.changePattern(0)
        self.nSteps(5*pat0NumInputs)
        self.InputCreator.changePattern(1)
        self.nSteps(5*pat1NumInputs)

        # Now run through the old pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.changePattern(0)
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        outputSDR00 = self.getColumnGridOutput(self.htm, 0, 0)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(pat0NumInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(pat0NumInputs)]
                                             for p in range(numPatternsTested)]
                                            )
        for i in range(pat0NumInputs):
            outputsFromPatternX[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        #import ipdb; ipdb.set_trace()

        self.InputCreator.changePattern(1)
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternX[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(0)
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        for i in range(pat0NumInputs):
            outputsFromPatternXAgain[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the second pattern.
        self.InputCreator.changePattern(1)
        # Restore all the outputs from the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternXAgain[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Now we need to compare the two outptus from both times the two input patterns
        # were stored. Also compare the output from the two times the different patterns were
        # recorded.
        similarOutputsIn1 = np.zeros(pat0NumInputs)
        similarOutputsIn2 = np.zeros(pat1NumInputs)
        similarOutputsIn1And2 = np.zeros(pat0NumInputs)
        for i in range(pat0NumInputs):
            similarOutputsIn1[i] = self.gridsSimilar(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            similarOutputsIn2[i] = self.gridsSimilar(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])
            similarOutputsIn1And2[i] = self.gridsSimilar(outputsFromPatternXAgain[0][i],
                                                         outputsFromPatternXAgain[1][i])

            # The two times the ouput from the htm was recorded for each pattern
            # the same output SDR should have been created.
            print "similarOutputsIn1[%s] = %s" % (i, similarOutputsIn1[i])
            print "similarOutputsIn2[%s] = %s" % (i, similarOutputsIn2[i])
            print "similarOutputsIn1And2[%s] = %s" % (i, similarOutputsIn1And2[i])

        print "Averaged similarOutputsIn1 = %s" % np.average(similarOutputsIn1)
        print "Averaged similarOutputsIn2 = %s" % np.average(similarOutputsIn2)
        print "Averaged similarOutputsIn1And2 = %s" % np.average(similarOutputsIn1And2)

        assert np.average(similarOutputsIn1) >= 0.95
        assert np.average(similarOutputsIn2) >= 0.95
        # The outputs from the two input sequences should be very similar.
        # The second pattern is the same as the first just with extra bits on.
        # They would be exactly the same but temporal pooling splits them up abit.
        assert np.average(similarOutputsIn1And2) >= 0.5

        #gui.startHtmGui(self.htm, self.InputCreator)

    def test_case2(self):
        '''
        Spatial pooler multiple input patterns.

        Same as test case1 only this time sequence A and B are similar
        but different dotted vertical lines moving left to right.
        The htm should be able to differentiate between the two.

        The output from the htm for both the input sequences should be different.
        This is because they both contain different features.

        '''
        # Create the input patterns to test with
        lineWidth = 1
        dottedLineHeight = 1
        patNumInputs = self.InputCreator.getNumInputsInSeq(0)
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth, dottedLineHeight)
        # The second pattern is just a dotted vertical line of height 2.
        lineWidth = 1
        dottedLineHeight = 2
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth, dottedLineHeight)

        # We will use these defined pattern above for testing.
        numPatternsTested = self.InputCreator.getNumCustomSequences()
        pat0NumInputs = self.InputCreator.getNumInputsInSeq(0)
        pat1NumInputs = self.InputCreator.getNumInputsInSeq(1)

        # Run the patterns through the htm multiple times
        # this is done so the htm can settle on a representation
        # for each input.
        self.InputCreator.changePattern(1)
        self.nSteps(5*pat0NumInputs)
        self.InputCreator.changePattern(2)
        self.nSteps(5*pat1NumInputs)

        # Now run through the old pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.changePattern(1)
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        outputSDR00 = self.getColumnGridOutput(self.htm, 0, 0)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(pat0NumInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(pat0NumInputs)]
                                             for p in range(numPatternsTested)]
                                            )
        for i in range(pat0NumInputs):
            outputsFromPatternX[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        #import ipdb; ipdb.set_trace()

        self.InputCreator.changePattern(2)
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternX[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(1)
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        for i in range(pat0NumInputs):
            outputsFromPatternXAgain[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the second pattern.
        self.InputCreator.changePattern(2)
        # Restore all the outputs from the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternXAgain[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Now we need to compare the two outptus from both times the two input patterns
        # were stored. Also compare the output from the two times the different patterns were
        # recorded.
        similarOutputsIn1 = np.zeros(pat0NumInputs)
        similarOutputsIn2 = np.zeros(pat1NumInputs)
        similarOutputsIn1And2 = np.zeros(pat0NumInputs)
        for i in range(pat0NumInputs):
            similarOutputsIn1[i] = self.gridsSimilar(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            similarOutputsIn2[i] = self.gridsSimilar(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])
            similarOutputsIn1And2[i] = self.gridsSimilar(outputsFromPatternXAgain[0][i],
                                                         outputsFromPatternXAgain[1][i])

            # The two times the ouput from the htm was recorded for each pattern
            # the same output SDR should have been created.
            print "similarOutputsIn1[%s] = %s" % (i, similarOutputsIn1[i])
            print "similarOutputsIn2[%s] = %s" % (i, similarOutputsIn2[i])
            print "similarOutputsIn1And2[%s] = %s" % (i, similarOutputsIn1And2[i])

        print "Averaged similarOutputsIn1 = %s" % np.average(similarOutputsIn1)
        print "Averaged similarOutputsIn2 = %s" % np.average(similarOutputsIn2)
        print "Averaged similarOutputsIn1And2 = %s" % np.average(similarOutputsIn1And2)

        assert np.average(similarOutputsIn1) >= 0.95
        assert np.average(similarOutputsIn2) >= 0.95
        # The outputs from the two input sequences should be very different.
        # The two dotted lines do not share similar features.
        assert np.average(similarOutputsIn1And2) < 0.05

        #gui.startHtmGui(self.htm, self.InputCreator)

    def test_case3(self):
        '''
        Spatial pooler multiple input patterns.

        Same as test case2.
        There are three different dotted verticla lines.

        The htm should be able to differentiate between the three.
        The output from the htm for all the input sequences should be different.
        This is because the three different sequences contain different features.

        '''
        # Create the input patterns to test with
        lineWidth = 1
        dottedLineHeight = 1
        patNumInputs = self.InputCreator.getNumInputsInSeq(0)
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth, dottedLineHeight)
        # The second pattern is just a dotted vertical line of height 2.
        lineWidth = 1
        dottedLineHeight = 2
        patNumInputs = self.InputCreator.getNumInputsInSeq(0)
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth, dottedLineHeight)
        # The third pattern is just a dotted vertical line of height 3.
        lineWidth = 1
        dottedLineHeight = 3
        dottedLineGap = 2
        patNumInputs = self.InputCreator.getNumInputsInSeq(0)
        self.createDottedVerticalLineSeq(patNumInputs, lineWidth, dottedLineHeight, dottedLineGap)

        # We will use these defined pattern above for testing.
        numPatternsTested = self.InputCreator.getNumCustomSequences()
        # Save the number of inputs for each pattern.
        pat0NumInputs = self.InputCreator.getNumInputsInSeq(0)
        pat1NumInputs = self.InputCreator.getNumInputsInSeq(1)
        pat2NumInputs = self.InputCreator.getNumInputsInSeq(2)

        # Run the patterns through the htm multiple times
        # this is done so the htm can settle on a representation
        # for each input.
        self.InputCreator.changePattern(1)
        self.nSteps(5*pat0NumInputs)
        self.InputCreator.changePattern(2)
        self.nSteps(5*pat1NumInputs)
        self.InputCreator.changePattern(3)
        self.nSteps(5*pat2NumInputs)

        # Create an array to store results for each output of the htm
        # for each input in each sequence.
        outputSDR00 = self.getColumnGridOutput(self.htm, 0, 0)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(pat0NumInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(pat0NumInputs)]
                                             for p in range(numPatternsTested)]
                                            )

        # Now run through the old pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output from the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.changePattern(1)
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        for i in range(pat0NumInputs):
            outputsFromPatternX[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(2)
        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternX[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(3)
        # Store the outputs from the third pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(pat2NumInputs)
        for i in range(pat2NumInputs):
            outputsFromPatternX[2][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(1)
        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(pat0NumInputs)
        for i in range(pat0NumInputs):
            outputsFromPatternXAgain[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the second pattern.
        self.InputCreator.changePattern(2)
        # Restore all the outputs from the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternXAgain[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the third pattern.
        self.InputCreator.changePattern(3)
        # Restore all the outputs from the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(pat1NumInputs)
        for i in range(pat1NumInputs):
            outputsFromPatternXAgain[2][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Now we need to compare the two outptus from both times the two input patterns
        # were stored. Also compare the output from the two times the different patterns were
        # recorded.
        similarOutputsIn1 = np.zeros(pat0NumInputs)
        similarOutputsIn2 = np.zeros(pat1NumInputs)
        similarOutputsIn3 = np.zeros(pat2NumInputs)
        similarOutputsIn1And2 = np.zeros(pat0NumInputs)
        similarOutputsIn1And3 = np.zeros(pat0NumInputs)
        similarOutputsIn2And3 = np.zeros(pat0NumInputs)
        for i in range(pat0NumInputs):
            similarOutputsIn1[i] = self.gridsSimilar(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            similarOutputsIn2[i] = self.gridsSimilar(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])
            similarOutputsIn3[i] = self.gridsSimilar(outputsFromPatternX[2][i],
                                                     outputsFromPatternXAgain[2][i])
            similarOutputsIn1And2[i] = self.gridsSimilar(outputsFromPatternXAgain[0][i],
                                                         outputsFromPatternXAgain[1][i])
            similarOutputsIn1And3[i] = self.gridsSimilar(outputsFromPatternXAgain[0][i],
                                                         outputsFromPatternXAgain[2][i])
            similarOutputsIn2And3[i] = self.gridsSimilar(outputsFromPatternXAgain[1][i],
                                                         outputsFromPatternXAgain[2][i])

            # The two times the ouput from the htm was recorded for each pattern
            # the same output SDR should have been created.
            print "similarOutputsIn1[%s] = %s" % (i, similarOutputsIn1[i])
            print "similarOutputsIn2[%s] = %s" % (i, similarOutputsIn2[i])
            print "similarOutputsIn3[%s] = %s" % (i, similarOutputsIn3[i])
            print "similarOutputsIn1And2[%s] = %s" % (i, similarOutputsIn1And2[i])
            print "similarOutputsIn1And3[%s] = %s" % (i, similarOutputsIn1And3[i])
            print "similarOutputsIn2And2[%s] = %s" % (i, similarOutputsIn2And3[i])

        print "Averaged similarOutputsIn1 = %s" % np.average(similarOutputsIn1)
        print "Averaged similarOutputsIn2 = %s" % np.average(similarOutputsIn2)
        print "Averaged similarOutputsIn3 = %s" % np.average(similarOutputsIn3)
        print "Averaged similarOutputsIn1And2 = %s" % np.average(similarOutputsIn1And2)
        print "Averaged similarOutputsIn1And3 = %s" % np.average(similarOutputsIn1And3)
        print "Averaged similarOutputsIn2And3 = %s" % np.average(similarOutputsIn2And3)

        assert np.average(similarOutputsIn1) >= 0.95
        assert np.average(similarOutputsIn2) >= 0.95
        # The outputs from the three input sequences should be very different.
        # The three different dotted lines do not share similar features.
        assert np.average(similarOutputsIn1And2) < 0.05
        assert np.average(similarOutputsIn1And3) < 0.05
        assert np.average(similarOutputsIn2And3) < 0.05

        #gui.startHtmGui(self.htm, self.InputCreator)


