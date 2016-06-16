from HTM_network import HTM_network
import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
from copy import deepcopy
from utilities import simpleVerticalLineInputs as svli, measureTemporalPooling as mtp
from utilities import sdrFunctions
from utilities import startHtmGui as gui


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
                                'maxNumTempPoolPatterns': 3,
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
                                'spatialPermanenceDec': 0.05,
                                'maxNumTempPoolPatterns': 3,
                                'permanenceInc': 0.1,
                                'permanenceDec': 0.02,
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
                                'inhibitionWidth': 20,
                                'inhibitionHeight': 2,
                                'centerPotSynapses': 1,
                                'connectPermanence': 0.3,
                                'potentialWidth': 30,
                                'potentialHeight': 10,
                                'spatialPermanenceInc': 0.2,
                                'spatialPermanenceDec': 0.05,
                                'maxNumTempPoolPatterns': 3,
                                'permanenceInc': 0.15,
                                'permanenceDec': 0.05,
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
        self.htm = HTM_network.HTM(self.InputCreator.createSimGrid(),
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

        gui.startHtmGui(self.htm, self.InputCreator)

        self.nSteps(400)

        tempPoolPercent = 0
        # Run through all the inputs twice and find the average temporal pooling percent
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            htmOutput = self.htm.levelOutput(self.numLevels-1)
            tempPoolPercent = self.temporalPooling.temporalPoolingPercent(htmOutput)
            print "Temporal pooling percent = %s" % tempPoolPercent

        #gui.startHtmGui(self.htm, self.InputCreator)

        # More then this percentage of temporal pooling should have occurred
        assert tempPoolPercent >= 0.75

    def test_case4(self):
        '''
        This test is designed to make sure that temporal pooling
        increase up the heirarchy of layers.
        '''
        #gui.startHtmGui(self.htm, self.InputCreator)
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

        #gui.startHtmGui(self.htm, self.InputCreator)

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

        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        tempPoolPercentPat1 = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercentPat1[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
        # Print the values of temporal pooling that occurred in each layer
        for i in range(len(tempPoolPercentPat1)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercentPat1[i])

        # Change the input pattern to pattern 1 (this is a pattern of right to left)
        self.InputCreator.changePattern(1)
        self.nSteps(400)
        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        tempPoolPercentPat2 = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercentPat2[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercentPat2)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercentPat2[i])

        # change the input pattern back to the original one to see if it has been
        # remebered and tmporal pooling still occurs.
        self.InputCreator.changePattern(0)
        self.InputCreator.setIndex(0)
        self.nSteps(self.InputCreator.numInputs)
        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        tempPoolPercentPat3 = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercentPat3[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercentPat3)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercentPat3[i])

        # change the input pattern back to the second pattern to see if it has been
        # remebered and tmporal pooling still occurs.
        self.InputCreator.changePattern(1)
        self.InputCreator.setIndex(0)
        self.nSteps(self.InputCreator.numInputs)
        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        tempPoolPercentPat4 = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercentPat4[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercentPat4)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercentPat4[i])

        # Now the temporal pooling calculated in tempPoolPercentPat1 should closely
        # match the temporal pooling occurring in tempPoolPercentPat3. Also
        # tempPoolPercentPat2 should match tempPoolPercentPat4.
        # The values should be within 5 percent of each other. They aren't exactly
        # the same because the first input after the pattern change is unexpected
        # and causes bursting in the network.
        for i in range(len(tempPoolPercentPat1)):
            assert(abs(tempPoolPercentPat1[i] - tempPoolPercentPat3[i]) <= 0.05)
            assert(abs(tempPoolPercentPat2[i] - tempPoolPercentPat4[i]) <= 0.05)

    def test_similarPatterns(self):
        '''
        This test is designed to make sure different temporal pooling
        patterns form even for input patterns that are similar and
        maybe contain some of the same inputs.

        TODO
        THIS TEST DOESN'T WORK YET!
        THE TEMPORAL POOLER NEEDS MODIFICATIONS!
        '''

        # Use a sequence of input patterns that uses every 2nd input of the
        # left to right line sequence pattern in the input creator class.
        # This is the left to right pattern sequence.
        self.InputCreator.changePattern(0)

        gui.startHtmGui(self.htm, self.InputCreator)
        self.nSteps(100)

        # # Measure the temporal pooling for each layer. This requires
        # # a temporal pooling measuring class per layer.
        # self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        # tempPoolPercentPat1 = [0 for i in range(self.numLayers)]
        # # Run through all the inputs twice and find the average temporal pooling percent
        # # for each of the layers
        # for i in range(2*self.InputCreator.numInputs):
        #     self.step()
        #     for layer in range(self.numLayers):
        #         gridOutput = self.htm.regionArray[0].layerOutput(layer)
        #         tempPoolPercentPat1[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
        # # Print the values of temporal pooling that occurred in each layer
        # for i in range(len(tempPoolPercentPat1)):
        #     print "layer %s temp pooling = %s" % (i, tempPoolPercentPat1[i])
        #     # Store the top most layers output.
        #     # This is used to compare to temporally pooled output
        #     # of the topmost layer after the second pattern has been learnt.
        #     topGridOutputPat1 = self.htm.regionArray[0].layerOutput(self.numLayers-1)

        self.InputCreator.changePattern(3)
        gui.startHtmGui(self.htm, self.InputCreator)
        self.nSteps(150)
        self.InputCreator.changePattern(0)
        gui.startHtmGui(self.htm, self.InputCreator)
        self.nSteps(150)
        self.InputCreator.changePattern(3)
        gui.startHtmGui(self.htm, self.InputCreator)

        # # Measure the temporal pooling for each layer. This requires
        # # a temporal pooling measuring class per layer.
        # self.temporalPoolingMeasures = [mtp.measureTemporalPooling() for i in range(self.numLayers)]
        # tempPoolPercentPat2 = [0 for i in range(self.numLayers)]
        # # Run through all the inputs twice and find the average temporal pooling percent
        # # for each of the layers
        # for i in range(2*self.InputCreator.numInputs):
        #     self.step()
        #     for layer in range(self.numLayers):
        #         gridOutput = self.htm.regionArray[0].layerOutput(layer)
        #         tempPoolPercentPat2[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
        #         # Less then this percentage of temporal pooling should have occurred
        # for i in range(len(tempPoolPercentPat2)):
        #     print "layer %s temp pooling = %s" % (i, tempPoolPercentPat2[i])
        #     topGridOutputPat2 = self.htm.regionArray[0].layerOutput(self.numLayers-1)

        # # Calcualte how similar the current output temporally pooled pattern
        # # is compared to the old one
        # tempPooledSimilarity = sdrFunctions.similarInputGrids(topGridOutputPat1, topGridOutputPat2)
        # print "The temporal pooled pattern is %s percent similar" % tempPooledSimilarity

        # self.InputCreator.changePattern(0)
        # self.nSteps(2*self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(2*self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # self.nSteps(2*self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(2)
        # self.nSteps(self.InputCreator.numInputs)
        # self.InputCreator.changePattern(0)
        # gui.startHtmGui(self.htm, self.InputCreator)

        # self.InputCreator.changePattern(2)
        # gui.startHtmGui(self.htm, self.InputCreator)

        # self.InputCreator.changePattern(0)
        # gui.startHtmGui(self.htm, self.InputCreator)

        # self.InputCreator.changePattern(2)
        # gui.startHtmGui(self.htm, self.InputCreator)

        #gui.startHtmGui(self.htm, self.InputCreator)




