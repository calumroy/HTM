#!/usr/bin/python

"""
A simple class to create an input grid for a HTM network.
This particular example creates a moving line that moves from side to side.

author: Calum Meiklejohn
website: calumroy.com
"""
import numpy as np
import math
import random


class InputCreator:

    def __init__(self, gridWidth, gridHeight, overlap):
        self.pos_x = int(gridWidth/2)     # The column number that the vertical line is at.
        self.direction = 1  # ! if the vertical line is moving right -1 if it's moving left.
        self.overlap = overlap  # The number of columns that the line can overlap either side.
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

    def newInput(self):
        # If the vertical line is at column zero or at the end column then reverse the direction.
        if self.pos_x >= self.gridWidth-1 and self.direction == 1:
            self.direction = -1
        if self.pos_x <= 0 and self.direction == -1:
            self.direction = 1
        # Move the vertical lines postion
        self.pos_x = self.pos_x + self.direction

        return self.createInput(self.pos_x, self.gridWidth, self.gridHeight, self.overlap)

    def createInput(self, pos_x, gridWidth, gridHeight, overlap):
        newInput = np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
        for row in range(gridHeight):
            for col in range(gridWidth):
                if col >= (round(pos_x-overlap)) and col <= (round(pos_x+overlap)):
                    newInput[row][col] = 1
        #print "grid = ",angleInput
        return newInput

