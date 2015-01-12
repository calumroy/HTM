

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
        self.memories = [self.command]
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
        # We add the new memory to the start of the
        # array then delete the memory at the end of the array.
        # All the memories should be in order from
        # most recent to oldest.
        if len(self.memories) < self.historyLength:
            self.memories = np.insert(self.memories, 0, memory)
        else:
            newArray = np.insert(self.memories, 0, memory)
            newArray = np.delete(newArray, len(newArray)-1)
            self.memories = newArray
            # Now check to see if a new command from the thalamus should be issued.
            self.reconsider()
