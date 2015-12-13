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
from utilities import startHtmGui as gui

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


class test_spatialPoolingSuite2:
    def setUp(self):
        '''
        We are tesing the spatial pooler, and it's
        ability to  work with the temporal pooler to remember multiple sequences.

        '''

        params = testParameters

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        self.numInputs = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
        inputHeight = 2*params['HTM']['columnArrayHeight']

        self.InputCreator = svli.simpleVerticalLineInputs(inputWidth, inputHeight, self.numInputs)

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
        Spatial pooler multiple input patterns.

        Test the spatial pooler and make sure that if it learns
        sequence A then another sequence B, A should not be forgotten.

        Two patterns are learnt by the htm.
        Once learnt the output SDR for each input in each pattern
        is stored. Then another pattern is sent as an input.
        After sometime the original patterns again become the input and
        new outputs are compared against the previously stored outputs
        to see if they are they same.

        Sequence A in this case is a vertical line that moves left to right.
        Sequence B is a vertical line that moves right to left.

        '''
        # How many patterns are we comparing against
        numPatternsTested = 2

        # Run the patterns through the htm multiple times
        # this is done so the htm can settle on a representation
        # for each input.
        self.InputCreator.changePattern(0)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(1)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(0)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(1)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(0)
        self.nSteps(self.numInputs)

        # Now run through the old pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output form the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        outputSDR00 = self.getColumnGridOutput(self.htm, 0, 0)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(self.numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(self.numInputs)]
                                             for p in range(numPatternsTested)]
                                            )

        for i in range(self.numInputs):
            outputsFromPatternX[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        #import ipdb; ipdb.set_trace()

        self.InputCreator.changePattern(1)

        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternX[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(0)

        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternXAgain[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the second pattern.
        self.InputCreator.changePattern(1)

        # Restore all the outputs. form the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternXAgain[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Now we need to compare the two ouptus from both times the two input patterns
        # were stored.
        similarOutputsIn1 = np.zeros(self.numInputs)
        similarOutputsIn2 = np.zeros(self.numInputs)
        for i in range(self.numInputs):
            similarOutputsIn1[i] = self.gridsSimilar(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            similarOutputsIn2[i] = self.gridsSimilar(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])

            print "similarOutputsIn1[%s] = %s" % (i, similarOutputsIn1[i])
            print "similarOutputsIn2[%s] = %s" % (i, similarOutputsIn2[i])

        assert np.average(similarOutputsIn1) >= 0.95
        assert np.average(similarOutputsIn2) >= 0.95

        #gui.startHtmGui(self.htm, self.InputCreator)

    def test_case2(self):
        '''
        Spatial pooler multiple input patterns.

        Similar to test case1. Only this time pattern B is very similar
        to pattern A. Test the spatial pooler and make sure that if it learns
        sequence A then another sequence B, A should not be forgotten.

        Two patterns are learnt by the htm.
        Once learnt the output SDR for each input in each pattern
        is stored. Then another pattern is sent as an input.
        After sometime the original patterns again become the input and
        new outputs are compared against the previously stored outputs
        to see if they are they same.

        Sequence A in this case is a vertical line that moves left to right.
        Sequence B is a vertical line that left to right also but jumps position.

        '''
        # How many patterns are we comparing against
        numPatternsTested = 2

        # Run the patterns through the htm multiple times
        # this is done so the htm can settle on a representation
        # for each input.
        self.InputCreator.changePattern(0)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(2)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(0)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(2)
        self.nSteps(5*self.numInputs)
        self.InputCreator.changePattern(0)
        self.nSteps(self.numInputs)

        # Now run through the old pattern and store each output SDR
        # These are used to compare against later on.
        # outputsFromPatternX is an array storing a list of SDRs representing
        # the output form the htm for every input in a particular pattern.
        # Reset the pattern to the start
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        outputSDR00 = self.getColumnGridOutput(self.htm, 0, 0)
        outputsFromPatternX = np.array([[np.zeros_like(outputSDR00)
                                         for n in range(self.numInputs)]
                                        for p in range(numPatternsTested)]
                                       )
        outputsFromPatternXAgain = np.array([[np.zeros_like(outputSDR00)
                                              for n in range(self.numInputs)]
                                             for p in range(numPatternsTested)]
                                            )

        for i in range(self.numInputs):
            outputsFromPatternX[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        #import ipdb; ipdb.set_trace()

        self.InputCreator.changePattern(2)

        # Store the outputs from the second pattern.
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternX[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        self.InputCreator.changePattern(0)

        # Now the pattern has been changed back to the first one.
        # Rerun through all the inputs and store the outputs from
        # the first pattern again.
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternXAgain[0][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Change the pattern back to the second pattern.
        self.InputCreator.changePattern(2)

        # Restore all the outputs. form the second pattern
        self.InputCreator.setIndex(0)
        self.nSteps(self.numInputs)
        for i in range(self.numInputs):
            outputsFromPatternXAgain[1][i] = self.getColumnGridOutput(self.htm, 0, 0)
            self.step()

        # Now we need to compare the two ouptus from both times the two input patterns
        # were stored.
        similarOutputsIn1 = np.zeros(self.numInputs)
        similarOutputsIn2 = np.zeros(self.numInputs)
        for i in range(self.numInputs):
            similarOutputsIn1[i] = self.gridsSimilar(outputsFromPatternX[0][i],
                                                     outputsFromPatternXAgain[0][i])
            similarOutputsIn2[i] = self.gridsSimilar(outputsFromPatternX[1][i],
                                                     outputsFromPatternXAgain[1][i])

            print "similarOutputsIn1[%s] = %s" % (i, similarOutputsIn1[i])
            print "similarOutputsIn2[%s] = %s" % (i, similarOutputsIn2[i])

        assert np.average(similarOutputsIn1) >= 0.95
        assert np.average(similarOutputsIn2) >= 0.95

        #gui.startHtmGui(self.htm, self.InputCreator)





