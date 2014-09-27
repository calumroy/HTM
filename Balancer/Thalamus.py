

"""
HTM GUI
author: Calum Meiklejohn
website: calumroy.com

This class is a simple thalamus class to be used by the HTM network.
The purpose of this class is to direct the HTM network to control the
ouptus such that desired input states are reached.

"""
import numpy as np
import random


class Thalamus:
    def __init__(self, columnArrayWidth, columnArrayHeight):
        # The thalamus contains a 'memory' variable whose purpose is to
        # store in an array memories which directed the HTM network outputs
        # to produce desired inputs.
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        # The thalamus command position.
        self.memPos = 0
        self.memory = np.array([[0 for i in range(self.width)]
                                for j in range(self.height)])

        # Store the history of the pendulums angle in this array.
        # This will be used to work out if a new command
        # should be sent to the HTM from the thalamus.
        self.historyLength = 10
        self.angleHistory = np.array([])

    def returnMemory(self):
        memWidth = self.width
        memHeight = self.height
        overlap = 2
        for row in range(memHeight):
            for col in range(memWidth):
                self.memory[row][col] = 0
                if col >= (round(self.memPos-overlap)) and col <= (round(self.memPos+overlap)):
                    self.memory[row][col] = 1
        return self.memory

    def reconsider(self):
        # Decide whether to change the current command based on the history of the
        # angle input form the simulator.
        angleCenterCount = 0
        if len(self.angleHistory) == self.historyLength:
            for i in self.angleHistory:
                if i == 0:
                    angleCenterCount += 1
            if angleCenterCount < 5:
                self.angleHistory = np.array([])
                self.changeMemPos(random.randint(0, self.width))

    def changeMemPos(self, newMemPos):
        self.memPos = newMemPos

    def addToHistory(self, angle):
         # We add the new angle to the start of the
        # array then delete the angle at the end of the array.
        # All the angles should be in order from
        # most recent to oldest.
        if len(self.angleHistory) < self.historyLength:
            self.angleHistory = np.insert(self.angleHistory, 0, angle)
        else:
            newArray = np.insert(self.angleHistory, 0, angle)
            newArray = np.delete(newArray, len(newArray)-1)
            self.angleHistory = newArray
            # Now check to see if a new command from the thalamus should be issued.
            self.reconsider()
