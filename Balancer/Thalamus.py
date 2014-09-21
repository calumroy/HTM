

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
    def __init__(self, columnArrayWidth, columnArrayHeight, cellsPerColumn):
        # The thalamus contains a 'memory' variable whose purpose is to
        # store in an array memories which directed the HTM network outputs
        # to produce desired inputs.
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.cellsPerColumn = cellsPerColumn
        # The thalamus command position.
        self.memPos = 0
        self.memory = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                                for j in range(self.height)])

    def returnMemory(self):
        memWidth = self.width*self.cellsPerColumn
        memHeight = self.height
        overlap = 2
        for row in range(memHeight):
            for col in range(memWidth):
                self.memory[row][col] = 0
                if col >= (round(self.memPos-overlap)) and col <= (round(self.memPos+overlap)):
                    self.memory[row][col] = 1
        return self.memory

    def changeMemPos(self, newMemPos):
        self.memPos = newMemPos