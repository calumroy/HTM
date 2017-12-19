import cProfile
import numpy as np
from utilities import sdrFunctions as SDRFunct
from HTM_layer import HTMLayer
from reinforcement_learning import Thalamus


class HTMRegion:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn, params):
        '''
        The HTMRegion is an object holding multiple HTMLayers. The region consists of
        simulated cortex layers.

        The lowest layer recieves the new input and feedback from the higher levels.

        The highest layer is a command/input layer. It recieves input from the
        lowest layers and from an outside thalamus class.
        This extra input is meant to direct the HTM.
        '''
        # The class contains multiple HTM layers stacked on one another
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.cellsPerColumn = cellsPerColumn
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        self.numLayers = params['numLayers']  # The number of HTM layers that make up a region.
        # An array to store the layers that make up this level (region).
        self.layerArray = np.array([], dtype=object)
        # Make a place to store the thalamus command.
        self.commandInput = np.array([[0 for i in range(self.width*cellsPerColumn)]
                                     for j in range(self.height)])
        # Setup space in the input for a command feedback SDR
        self.enableCommandFeedback = params['enableCommandFeedback']

        self.setupLayers(input, params['HTMLayers'])

        # create and store a thalamus class if the
        # command feedback param is true.
        self.thalamus = None
        if self.enableCommandFeedback == 1:
            # The width of the thalamus should match the width of the input grid.
            thalamusParams = params['Thalamus']
            self.thalamus = Thalamus.Thalamus(self.width*self.cellsPerColumn,
                                              self.height,
                                              thalamusParams)

    def setupLayers(self, input, htmLayerParams):
        # Set up the inputs to the HTM layers.
        # Note the params comes in a list of dics, one for each layer.
        # Layer 0 gets the new input.
        bottomLayerParams = htmLayerParams[0]
        self.layerArray = np.append(self.layerArray, HTMLayer(input,
                                                              self.width,
                                                              self.height,
                                                              self.cellsPerColumn,
                                                              bottomLayerParams))
        # The higher layers receive the lower layers output.
        for i in range(1, self.numLayers):
            # Try to get the parameters for this layer.
            layersParams = self.getLayerParams(htmLayerParams, i)

            lowerOutput = self.layerArray[i-1].output

            # The highest layer receives the lower layers input and
            # an input from the thalamus equal in size to the lower layers input.
            highestLayer = self.numLayers - 1

            if i == highestLayer and self.enableCommandFeedback == 1:
                lowerOutput = SDRFunct.joinInputArrays(self.commandInput, lowerOutput)

            self.layerArray = np.append(self.layerArray,
                                        HTMLayer(lowerOutput,
                                                 self.width,
                                                 self.height,
                                                 self.cellsPerColumn,
                                                 layersParams))

    def getLayerParams(self, params, layerIndex):
        # Get the parameters from the layer with the given index from this region
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        layerParams = None

        if len(params) >= layerIndex:
            layerParams = params[layerIndex]
        else:
            # Just return the parameters of the highest layer.
            layerParams = params[-1]

        return layerParams

    def updateThalamus(self):
        # Update the thalamus.
        # This updates the command input that comes from
        # the thalamus.
        # Get the predicted command from the command space.
        # Pass this to the thalamus
        if self.thalamus is not None:
            topLayer = self.numLayers-1
            predCommGrid = self.layerPredCommandOutput(topLayer)
            # print "predCommGrid = %s" % predCommGrid
            thalamusCommand = self.thalamus.pickCommand(predCommGrid)

            # Update each level of the htm with the thalamus command
            self.updateCommandInput(thalamusCommand)

    def rewardThalamus(self, reward):
        # Reward the Thalamus.
        if self.thalamus is not None:
            self.thalamus.rewardThalamus(reward)

    def updateCommandInput(self, newCommand):
        # Update the command input for the level
        # This input is used by the top layer in this level.
        self.commandInput = newCommand

    def updateRegionInput(self, input):
        # Update the input and outputs of the layers.
        # Layer 0 receives the new input.
        highestLayer = self.numLayers - 1

        self.layerArray[0].updateInput(input)

        # The middle layers receive inputs from the lower layer outputs
        for i in range(1, self.numLayers):
            lowerOutput = self.layerArray[i - 1].output
            # The highest layer receives the lower layers input and
            # the commandInput for the level, equal in size to the lower layers input.
            if i == highestLayer and self.enableCommandFeedback == 1:
                lowerOutput = SDRFunct.joinInputArrays(self.commandInput, lowerOutput)
            self.layerArray[i].updateInput(lowerOutput)

    def layerOutput(self, layer):
        # Return the output for the given layer.
        return self.layerArray[layer].output

    def regionOutput(self):
        # Return the output from the entire region.
        # This will be the output from the highest layer.
        highestLayer = self.numLayers - 1
        return self.layerOutput(highestLayer)

    def commandSpaceOutput(self, layer):
        # Return the output from the command space
        # This is the top half of the output from the selected layer
        layerHeight = self.layerArray[layer].height
        wholeOutput = self.layerArray[layer].output
        halfLayerHeight = int(layerHeight/2)

        commSpaceOutput = wholeOutput[0:halfLayerHeight, :]
        return commSpaceOutput

    def layerPredCommandOutput(self, layer):
        # Return the given layers predicted command output.
        # This is the predictive cells from the command space.
        # The command space is assumed to be the top half of the
        # columns.

        # Divide the predictive cell grid into two and take the
        # top most part which is the command space.
        totalPredGrid = self.layerArray[layer].predictiveCellGrid()
        # Splits the grid into 2 parts of equal or almost equal size.
        # This splits the top and bottom. Return the top at index[0].
        PredGridOut = np.array_split(totalPredGrid, 2)[0]

        return PredGridOut

    def spatialTemporal(self):
        i = 0
        for layer in self.layerArray:
            # print "     Layer = %s" % i
            i += 1
            layer.timeStep = layer.timeStep+1
            # Update the current layers input with the new input
            # This updates the spatial pooler
            layer.Overlap()
            layer.inhibition(layer.timeStep)
            layer.spatialLearning()
            # This updates the sequence pooler
            layer.sequencePooler(layer.timeStep)
            # This updates the temporal pooler
            layer.temporalPooler(layer.timeStep)
            # Update the output grid for the layer.
            layer.updateOutput()