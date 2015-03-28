from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM, HTMLayer, HTMRegion, Column
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import random
from copy import deepcopy


class simpleVerticalLineInputs:
    def __init__(self, width, height, numInputs):
        # The number of inputs to store
        self.numInputs = numInputs
        self.width = width
        self.height = height
        self.inputs = np.array([[[0 for i in range(self.width)]
                                for j in range(self.height)] for k in range(self.numInputs)])
        self.setInputs(self.inputs)
        # Use an index to keep track of which input to send next
        self.index = 0
        # A variable speifying the amount of noise in the inputs 0 to 1
        self.noise = 0.0
        # A variable indicating the chance that the next input is a random input from the sequence.
        # This variable is used to create an input sequence that sometimes changes. It is the probablity
        # that the next input is the correct input in the sequence
        self.sequenceProbability = 1.0

    def setInputs(self, inputs):
        # Will will create vertical lines in the input that move from side to side.
        # These inputs should then be used to test temporal pooling.
        for n in range(len(inputs)):
            for y in range(len(inputs[0])):
                for x in range(len(inputs[n][y])):
                    if x == n:
                        inputs[n][y][x] = 1

    def step(self, cellGrid):
        # Required function for a InputCreator class
        pass

    def createSimGrid(self):
        # Required function for a InputCreator class
        newGrid = None
        # Add some random noise to the next input
        # The next input is at the self.index
        if self.noise > 0.0:
            # Return a new grid so the original input is not over written.
            newGrid = deepcopy(self.inputs[self.index])

            for y in range(len(newGrid[0])):
                for x in range(len(newGrid[y])):
                    if random.random() < self.noise:
                        newGrid[y][x] = 1
        # Give the next outpu a chance to be an out of sequence input.
        if (random.random() < self.sequenceProbability):
            outputGrid = self.inputs[self.index]
        else:
            sequenceLen = len(self.inputs)
            outputGrid = self.inputs[random.randint(0, sequenceLen-1)]
        # Increment the index for next time
        self.index += 1
        if (self.index >= len(self.inputs)):
            self.index = 0
        # If noise was added return the noisy grid.
        if newGrid is not None:
            return newGrid
        else:
            return outputGrid


class measureTemporalPooling:
    '''
    The purpose of this class is to measure the amount of temporal pooling
    occuring across a set of input grids. This means measure the amount that
    the input grids change by.

    This class stores the input grid it receives.
    It then uses this to compare to future grid arrays.
    It creates a running average of how much each successive
    grid changes from the previous one.
    '''
    def __init__(self):
        self.grid = None
        # A running average totalling the percent of temporal pooling.
        self.temporalAverage = 0
        self.numInputGrids = 0

    def temporalPoolingPercent(self, grid):
        if self.grid is not None:
            totalPrevActiveIns = np.sum(self.grid != 0)
            totalAndGrids = np.sum(np.logical_and(grid != 0, self.grid != 0))
            percentTemp = float(totalAndGrids) / float(totalPrevActiveIns)
            #print "         totalAndGrids = %s" % totalAndGrids
            self.temporalAverage = (float(percentTemp) +
                                    float(self.temporalAverage*(self.numInputGrids-1)))/float(self.numInputGrids)
            #print "         percentTemp = %s" % percentTemp
        self.grid = deepcopy(grid)
        self.numInputGrids += 1
        return self.temporalAverage


class test_TemporalPooling:
    def setUp(self):
        '''
        This test will use multiple regions in one level.
        This is because the regions ouputs simply pass onto the next,
        their is no compilcated feedback happening with one level.

        To test the temporal pooling ability of the regions a sequence
        of inputs are repeatably inputted to the HTM. If after a number
        of steps the top most layer is only changing slightly compared to
        the bottom layer then temporal pooling is occuring.
        '''
        self.width = 10
        self.height = 30
        self.cellsPerColumn = 3
        self.numLevels = 1
        self.numLayers = 3

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        numInputs = self.width*self.cellsPerColumn
        inputWidth = self.width*self.cellsPerColumn
        inputHeight = 2*self.height

        self.InputCreator = simpleVerticalLineInputs(inputWidth, inputHeight, numInputs)
        #self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)
        self.htm = HTM(self.numLevels,
                       self.InputCreator.createSimGrid(),
                       self.width,
                       self.height,
                       self.cellsPerColumn,
                       self.numLayers)

        # Setup some parameters of the HTM
        self.setupParameters()

        # Measure the temporal pooling
        self.temporalPooling = measureTemporalPooling()

    def setupParameters(self):
        # Setup some parameters of the HTM
        self.htm.regionArray[0].layerArray[1].desiredLocalActivity = 4
        self.htm.regionArray[0].layerArray[2].desiredLocalActivity = 4
        #self.htm.regionArray[1].layerArray[0].desiredLocalActivity = 4
        #self.htm.regionArray[1].layerArray[0].changeColsPotRadius(4)
        #self.htm.regionArray[1].layerArray[1].desiredLocalActivity = 4

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
            htmOutput = self.htm.levelCommandOutput(self.numLevels-1)
            tempPoolPercent = self.temporalPooling.temporalPoolingPercent(htmOutput)
            print "Temporal pooling percent = %s" % tempPoolPercent

        # More then this percentage of temporal pooling should have occurred
        assert tempPoolPercent >= 0.8

    def test_case2(self):
        '''
        This test is designed to make sure that not much
        temporal pooling occurs for an input sequence that is
        changing constantly.
        '''
        self.nSteps(100)

        # Not much temporal pooling should occur for a sequence of random inputs.
        # Set the probabiltiy that the next input is in sequence to really low
        self.InputCreator.sequenceProbability = 0.0
        # Run through all the inputs twice and find the average temporal pooling percent
        for i in range(2*self.InputCreator.numInputs):
            self.step()
            htmOutput = self.htm.levelCommandOutput(self.numLevels-1)
            tempPoolPercent = self.temporalPooling.temporalPoolingPercent(htmOutput)
            print "Temporal pooling percent = %s" % tempPoolPercent

        #app = QtGui.QApplication(sys.argv)
        #self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        #sys.exit(app.exec_())

        # Less then this percentage of temporal pooling should have occurred
        assert tempPoolPercent < 0.2

    def test_case3(self):
        '''
        This test is designed to make sure that
        temporal pooling still occurs even when some inputs are missing
        '''
        self.nSteps(100)

        # Run through all the inputs twice and find the average temporal pooling percent
        for i in range(2*self.InputCreator.numInputs):
            # Only update every second input
            if (i % 2 == 0):
                newInput = self.InputCreator.createSimGrid()
                newInput = self.InputCreator.createSimGrid()
            self.htm.spatialTemporal(newInput)
            htmOutput = self.htm.levelCommandOutput(self.numLevels-1)
            tempPoolPercent = self.temporalPooling.temporalPoolingPercent(htmOutput)
            print "Temporal pooling percent = %s" % tempPoolPercent

        #app = QtGui.QApplication(sys.argv)
        #self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        #sys.exit(app.exec_())

        # More then this percentage of temporal pooling should have occurred
        assert tempPoolPercent > 0.6

    def test_case4(self):
        '''
        This test is designed to make sure that temporal pooling
        increase up the heirarchy of layers.
        '''
        self.nSteps(400)

        # Measure the temporal pooling for each layer. This requires
        # a temporal pooling measuring class per layer.
        self.temporalPoolingMeasures = [measureTemporalPooling() for i in range(self.numLayers)]

        tempPoolPercent = [0 for i in range(self.numLayers)]
        # Run through all the inputs twice and find the average temporal pooling percent
        # for each of the layers

        for i in range(2*self.InputCreator.numInputs):
            self.step()
            for layer in range(self.numLayers):
                gridOutput = self.htm.regionArray[0].layerOutput(layer)
                tempPoolPercent[layer] = self.temporalPoolingMeasures[layer].temporalPoolingPercent(gridOutput)
                print "Layer %s Temporal pooling percent = %s" % (layer, tempPoolPercent[layer])

        #app = QtGui.QApplication(sys.argv)
        #self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        #sys.exit(app.exec_())

        # Less then this percentage of temporal pooling should have occurred
        for i in range(len(tempPoolPercent)):
            print "layer %s temp pooling = %s" % (i, tempPoolPercent[i])
            if (i > 0):
                assert tempPoolPercent[i] > tempPoolPercent[i-1]

    def test_case5(self):
        '''
        This is a sample test
        '''
        app = QtGui.QApplication(sys.argv)
        self.htmGui = GUI_HTM.HTMGui(self.htm, self.InputCreator)
        sys.exit(app.exec_())

        assert 1 == 1
