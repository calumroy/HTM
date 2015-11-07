from HTM_Balancer import HTM
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli, measureTemporalPooling as mtp
import sys
import json

testParameters = {
                    'HTM':
                        {
                        'numLevels': 1,
                        'columnArrayWidth': 10,
                        'columnArrayHeight': 30,
                        'cellsPerColumn': 3,

                        'HTMRegions': [{
                            'numLayers': 3,
                            'enableHigherLevFb': 0,
                            'enableCommandFeedback': 0,

                            'HTMLayers': [{
                                'desiredLocalActivity': 2,
                                'inhibitionWidth': 4,
                                'inhibitionHeight': 3,
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
                                    'potentialWidth': 4,
                                    'potentialHeight': 4,
                                    'spatialPermanenceInc': 0.1,
                                    'spatialPermanenceDec': 0.02,
                                    'permanenceInc': 0.1,
                                    'permanenceDec': 0.02,
                                    'minDutyCycle': 0.01,
                                    'boostStep': 0,
                                    'historyLength': 2
                                }]
                            },
                            {
                                'desiredLocalActivity': 1,
                                'inhibitionWidth': 8,
                                'inhibitionHeight': 3,
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
                                    'potentialWidth': 8,
                                    'potentialHeight': 8,
                                    'spatialPermanenceInc': 0.1,
                                    'spatialPermanenceDec': 0.02,
                                    'permanenceInc': 0.1,
                                    'permanenceDec': 0.02,
                                    'minDutyCycle': 0.01,
                                    'boostStep': 0,
                                    'historyLength': 2
                                }]
                            },
                            {
                                'desiredLocalActivity': 1,
                                'inhibitionWidth': 20,
                                'inhibitionHeight': 3,
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
                                    'minOverlap': 2,
                                    'boost': 1,
                                    'potentialWidth': 30,
                                    'potentialHeight': 10,
                                    'spatialPermanenceInc': 0.2,
                                    'spatialPermanenceDec': 0.005,
                                    'permanenceInc': 0.1,
                                    'permanenceDec': 0.005,
                                    'minDutyCycle': 0.01,
                                    'boostStep': 0,
                                    'historyLength': 2
                                }]
                            }]
                        }]
                    }
                }


class test_temporalPoolingSuite2:
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
        self.htm = HTM(self.InputCreator.createSimGrid(),
                       params
                       )

        # Measure the temporal pooling
        self.temporalPooling = mtp.measureTemporalPooling()

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

    def test_case1(self):
        '''
        This test is designed to make sure that a minimum amount
        of temporal pooling occurs for a repeating input sequence.
        '''
        self.nSteps(400)

        tempPoolPercent = 0
        # Run through all the inputs twice and find the average temporal pooling percent
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            htmOutput = self.htm.levelOutput(self.numLevels-1)
            tempPoolPercent = self.temporalPooling.temporalPoolingPercent(htmOutput)
            print "Temporal pooling percent = %s" % tempPoolPercent

        # app = QtGui.QApplication(sys.argv)
        # self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        # sys.exit(app.exec_())

        # More then this percentage of temporal pooling should have occurred
        assert tempPoolPercent >= 0.75

    def test_case4(self):
        '''
        This test is designed to make sure that temporal pooling
        increase up the heirarchy of layers.
        '''
        self.nSteps(400)

        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]

        tempPoolPercent = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers

        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercent[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                #print "Layer %s Temporal pooling percent = %s" % (layer, tempPoolPercent[layer])

        # app = QtGui.QApplication(sys.argv)
        # self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        # sys.exit(app.exec_())

        # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercent)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercent[i])
            if (i > 0):
                assert tempPoolPercent[i] > tempPoolPercent[i-1]

    def test_multiPattern(self):
        '''
        This test is designed to make sure that temporal pooling
        occurs for multiple dofferent input patterns and the temporal pooling can
        still differentiate between them.

        Two patterns fed into the network and learnt. Different temporal pooling
        should occur for both the patterns.
        '''
        self.nSteps(400)

        # Change the input pattern to pattern 1 (this is a pattern of right to left)
        self.InputCreator.changePattern(1)

        self.nSteps(400)

        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]

        tempPoolPercent = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers

        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercent[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                #print "Layer %s Temporal pooling percent = %s" % (layer, tempPoolPercent[layer])

        # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercent)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercent[i])
            if (i > 0):
                assert tempPoolPercent[i] > tempPoolPercent[i-1]

        self.InputCreator.changePattern(0)

        app = QtGui.QApplication(sys.argv)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        sys.exit(app.exec_())
