# Title: HTM
# Description: HTM network
# Author: Calum Meiklejohn
# Development phase: alpha

import cProfile
import numpy as np
import copy
from utilities import sdrFunctions as SDRFunct

from HTM_region import HTMRegion


# Profiling function
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


class HTM:
    #@do_cprofile  # For profiling
    def __init__(self, input, params):

        # This class contains multiple HTM levels stacked on one another
        self.numLevels = params['HTM']['numLevels']   # The number of levels in the HTM network
        self.width = params['HTM']['columnArrayWidth']
        self.height = params['HTM']['columnArrayHeight']
        self.cellsPerColumn = params['HTM']['cellsPerColumn']
        self.regionArray = np.array([], dtype=object)

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        self.setupRegions(input, params)

        # create a place to store layers so they can be reverted.
        #self.HTMOriginal = copy.deepcopy(self.regionArray)
        self.HTMOriginal = None

    def setupRegions(self, input, htmParams):
        # Set up the HTM regions.
        # Note the region parameters comes in a list of dics, one for each region.
        # Setup the inputs and outputs between levels
        # Each regions input needs to make room for the command
        # feedback from another layer in possibly another level.
        htmRegionParams = htmParams['HTM']['HTMRegions']
        # The levels get inputs from the lower levels or the input to the entire network.
        for i in range(0, self.numLevels):
            # Try to get the parameters for this region else use the
            # last specified params from the highest region.
            if len(htmRegionParams) >= i+1:
                regionsParam = htmRegionParams[i]
            else:
                regionsParam = htmRegionParams[-1]
            # Get the parameters of the lowest layer in the bottom region.
            lowestLayersParams = self.getLayerParams(htmParams, 0, 0)
            # Get the output of the lower level to pass to the higher levels.
            # If this is the lowest then use the input to the HTM network.
            lowerOutput = None
            if i == 0:
                lowerOutput = input
            else:
                lowerOutput = self.regionArray[i-1].regionOutput()
            # If the region has higherLevFb param enabled add extra space to the input.
            if lowestLayersParams['enableFeedback'] == 1:
                # find out how much extra space is needed to add to the input grid for the feedback.
                lowLevelInd = lowestLayersParams['feedbackLevelInd']
                lowLayerInd = lowestLayersParams['feedbackLayerInd']
                fbParams = self.getLayerParams(htmParams, lowLevelInd, lowLayerInd)
                #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

                fbWidth = fbParams['columnArrayWidth']
                fbHeight = fbParams['columnArrayHeight']
                fbCellsPerCol = fbParams['cellsPerColumn']
                commandFeedback = np.array([[0 for i in range(fbWidth*fbCellsPerCol)]
                                            for j in range(int(fbHeight))])

                newInput = SDRFunct.joinInputArrays(commandFeedback, lowerOutput)
                #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
            else:
                newInput = lowerOutput

            self.regionArray = np.append(self.regionArray,
                                         HTMRegion(newInput,
                                                   self.width,
                                                   self.height,
                                                   self.cellsPerColumn,
                                                   regionsParam)
                                         )

    def getLayerParams(self, params, levelIndex, layerIndex):
        # Get the parameters from the layer with the given index in the given level
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        layerParams = None
        if len(params['HTM']['HTMRegions']) >= levelIndex:
            if len(params['HTM']['HTMRegions'][levelIndex]['HTMLayers']) >= layerIndex:
                layerParams = params['HTM']['HTMRegions'][levelIndex]['HTMLayers'][layerIndex]
            else:
                # Just return the parameters of the highest layer.
                layerParams = params['HTM']['HTMRegions'][levelIndex]['HTMLayers'][-1]
        else:
            print "\n    ERROR The feedback level parameters dont exist!"
        return layerParams

    def saveRegions(self):
        # Save the HTM so it can be reloaded.
        print "\n    SAVE COMMAND SYN "
        self.HTMOriginal = copy.deepcopy(self.regionArray)

    def loadRegions(self):
        # Save the synapses for the command area so they can be reloaded.
        if self.HTMOriginal is not None:
            print "\n    LOAD COMMAND SYN "
            self.regionArray = self.HTMOriginal
            # Need create a new deepcopy of the original
            self.HTMOriginal = copy.deepcopy(self.regionArray)
        # return the pointer to the HTM so the GUI can use it to point
        # to the correct object.
        return self.regionArray

    def updateTopLevelFb(self, newCommand):
        # Update the top level feedback command with a new one.
        self.topLevelFeedback = newCommand

    def updateHTMInput(self, input):
        # Update the input and outputs of the levels.
        # Level 0 receives the new input. The higher levels
        # receive inputs from the lower levels outputs

        # The input must also include the
        # command feedback from the higher layers.
        commFeedbackLev1 = np.array([])

        ### LEVEL 0 Update

        # The lowest levels lowest layer gets this new input.
        # All other levels and layers get inputs from lower levels and layers.
        if self.regionArray[0].layerArray[0].enableFeedback == 1:
            fbLevelInd = self.regionArray[0].layerArray[0].feedbackLevelInd
            fbLayerInd = self.regionArray[0].layerArray[0].feedbackLayerInd

            commFeedbackLev1 = self.levelLayerOutput(fbLevelInd, fbLayerInd)

        newInput = SDRFunct.joinInputArrays(commFeedbackLev1, input)
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        self.regionArray[0].updateRegionInput(newInput)

        ### HIGHER LEVELS UPDATE

        # Update each levels input. Combine the command feedback to the input.
        for i in range(1, self.numLevels):
            commFeedbackLevN = np.array([])
            lowerLevel = i-1
            # Set the output of the lower level
            highestLayer = self.regionArray[lowerLevel].numLayers - 1
            lowerLevelOutput = self.regionArray[lowerLevel].layerOutput(highestLayer)
            if self.regionArray[i].layerArray[0].enableFeedback == 1:
                fbLevelInd = self.regionArray[0].layerArray[0].feedbackLevelInd
                fbLayerInd = self.regionArray[0].layerArray[0].feedbackLayerInd

                commFeedbackLevN = self.levelLayerOutput(fbLevelInd, fbLayerInd)

            # Update the newInput for the current level in the HTM
            newInput = SDRFunct.joinInputArrays(commFeedbackLevN, lowerLevelOutput)
            self.regionArray[i].updateRegionInput(newInput)

    def levelLayerOutput(self, levelIndex, layerIndex):
        # Return the output from the desired layer in the desired level.
        highestLevel = self.numLevels-1
        highestLayer = self.regionArray[levelIndex].numLayers-1
        # Check that the desired level and layer index for feedback exists.
        assert highestLevel >= levelIndex
        assert highestLayer >= layerIndex

        return self.regionArray[levelIndex].layerOutput(layerIndex)

        #return self.regionArray[level].commandSpaceOutput(highestLayer)
        #return self.regionArray[level].regionOutput()

    def updateAllThalamus(self):
        # Update all the thalaums classes in each region
        for i in range(self.numLevels):
            self.regionArray[i].updateThalamus()

    def rewardAllThalamus(self, reward):
        # Reward the thalamus classes in each region
        for i in range(self.numLevels):
            self.regionArray[i].rewardThalamus(reward)

    #@do_cprofile  # For profiling
    def spatialTemporal(self, input):
        # Update the spatial and temporal pooler.
        # Find spatial and temporal patterns from the input.
        # This updates the columns and all their vertical
        # synapses as well as the cells and the horizontal Synapses.
        # Update the current levels input with the new input
        self.updateHTMInput(input)
        i = 0
        for level in self.regionArray:
            #print "Level = %s" % i
            i += 1
            level.spatialTemporal()




