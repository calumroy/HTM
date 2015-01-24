

"""
HTM GUI
author: Calum Meiklejohn
website: calumroy.com

This class is a simple thalamus class to be used by the HTM network.
The purpose of this class is to direct the HTM network to control the
outputs such that desired input states are reached.

"""
import numpy as np
import random


class Thalamus:
    def __init__(self, columnArrayWidth, columnArrayHeight):
        # The thalamus contains a 'memories' variable whose purpose is to
        # store in an array of memory grids which directed the HTM network outputs
        # to produce desired inputs.
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.command = np.array([[0 for i in range(self.width)]
                                for j in range(self.height)])
        self.memories = np.array([self.command])
        self.historyLength = 5
        # store the input angles
        self.angleHistory = []

    def returnMemory(self):
        return self.memories[0]

    def reconsider(self):
        # Decide whether to change the current command based on the history of the
        # angle input from the simulator.
        angleCenterCount = 0
        if len(self.angleHistory) == self.historyLength:
            for i in self.angleHistory:
                if i == 0:
                    angleCenterCount += 1
            if angleCenterCount < 5:
                self.angleHistory = np.array([])
                self.changeMemPos(random.randint(0, self.width))

    def addToHistory(self, memory):
        # We add the new memory to the end of the
        # array then delete the memory at the start of the array.
        # All the memories should be in order from
        # oldest to most recent.
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        if len(self.memories) < self.historyLength:
            # add the new 2d array to the array of 2d memories
            # The axis must be specified otherwise the array is flattened
            if (self.checkArraySizesMatch(self.memories[0], memory) is True):
                self.memories = np.append(self.memories, [memory], axis=0)
            else:
                print "Error: The new memory size is different to the previous ones"
        else:
            # Axis must be specified and the memory must be enclosed in []
            # Without this the memory will not be appended in whole to memories.
            newArray = np.append(self.memories, [memory], axis=0)
            # Delete the oldest memory at the start of the memories array
            newArray = np.delete(newArray, 0, 0)
            self.memories = newArray
            # Now check to see if a new command from the thalamus should be issued.
            self.reconsider()

    def checkArraySizesMatch(self, array1, array2):
        # Check if arrays up to dimension N are of equal size
        subArray1 = array1
        subArray2 = array2
        while (type(subArray1).__name__ == 'ndarray' and
               type(subArray2).__name__ == 'ndarray'):
            if (len(subArray1) != len(subArray2)):
                # arrays are not equal size
                return False
            if ((len(subArray1) == 0 and len(subArray2) != 0) or (
                len(subArray2) == 0 and len(subArray1) != 0)):
                # arrays are not equal size
                return False
            if (len(subArray1) == 0 and len(subArray2) == 0):
                # arrays are equal break to return true
                break
            # Check that the next dimension is still an array
            if (type(subArray1[0]).__name__ == 'ndarray' and
                type(subArray2[0]).__name__ == 'ndarray'):
                subArray1 = subArray1[0]
                subArray2 = subArray2[0]
            else:
                break
        return True

