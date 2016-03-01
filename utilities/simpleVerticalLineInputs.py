import numpy as np
from copy import deepcopy
import random
from utilities import sdrFunctions as SDRFunct


class simpleVerticalLineInputs:
    '''
    A class used to create different input sequences consisting of
    straight vertical lines. A number of different sequences are stored in
    an array of an array of matricies.

    The different sequences are defined in the setinputs method.
    To add a new sequence, increment the self.numPatterns and add
    the sequence to the end of the current self.inputs array.

    '''
    def __init__(self, width, height, numInputs):
        # The number of inputs to store
        self.numInputs = numInputs
        self.width = width
        self.height = height
        # How many input patterns to store
        self.numPatterns = 6
        # An index indicating the current pattern that is being used as a serias of input grids.
        self.patIndex = 0
        # An array storing different input patterns
        # Each pattern is a series of 2dArray grids storing binary patterns.
        self.inputs = np.array([[[[0 for i in range(self.width)]
                                for j in range(self.height)]
                                for k in range(self.numInputs)]
                                for l in range(self.numPatterns)])
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
        # This will create vertical lines in the input that move from side to side.
        # These inputs should then be used to test temporal pooling.
        # The first input pattern moves left to right
        for n in range(len(inputs[0])):
            for y in range(len(inputs[0][0])):
                for x in range(len(inputs[0][n][y])):
                    if x == n:
                        inputs[0][n][y][x] = 1
        # The second input pattern moves right to left
        for n in range(len(inputs[1])):
            for y in range(len(inputs[1][0])):
                for x in range(len(inputs[0][n][y])):
                    # reverse the pattern
                    if x == (len(inputs[1]) - 1 - n):
                        inputs[1][n][y][x] = 1
        # The third pattern is just every second input of the first pattern
        patIndex = 0
        for n in range(len(inputs[2])):
            patIndex = patIndex + 4
            if patIndex >= self.numInputs:
                patIndex = 0
            inputs[2][n] = inputs[0][patIndex]
        # The forth pattern is just every second input of the second pattern
        patIndex = 0
        for n in range(len(inputs[3])):
            patIndex = patIndex + 4
            if patIndex >= self.numInputs:
                patIndex = 0
            inputs[3][n] = inputs[1][patIndex]
        # The fifth pattern is the third pattern then the forth pattern
        patIndex = 0
        for n in range(len(inputs[3])):
            patIndex = patIndex + 1
            if patIndex >= self.numInputs:
                patIndex = 0
            if patIndex <= int(self.numInputs/2):
                inputs[4][n] = inputs[2][patIndex]
            else:
                inputs[4][n] = inputs[3][patIndex]
        # The 6th pattern is the first combined with the third pattern
        # by a logical or operation.
        patIndex = 0
        for n in range(len(inputs[3])):
            patIndex = patIndex + 1
            if patIndex >= self.numInputs:
                patIndex = 0
            inputs[5][n] = SDRFunct.orSDRPatterns(inputs[0][patIndex], inputs[2][patIndex])

    def changePattern(self, patternindex):
        # Change the input pattern
        self.patIndex = patternindex

    def step(self, cellGrid):
        # Required function for a InputCreator class
        pass

    def getReward(self):
        # Required function for a InputCreator class
        reward = 0
        return reward

    def setIndex(self, newIndex):
        # Change the index keeping track of where in a pattern the input
        # sequence is up to?
        self.index = newIndex

    def createSimGrid(self):
        # Required function for a InputCreator class
        newGrid = None
        # Add some random noise to the next input
        # The next input is at the self.index
        if self.noise > 0.0:
            # Return a new grid so the original input is not over written.
            newGrid = deepcopy(self.inputs[self.patIndex][self.index])

            for y in range(len(newGrid[0])):
                for x in range(len(newGrid[y])):
                    if random.random() < self.noise:
                        newGrid[y][x] = 1
        # Give the next outpu a chance to be an out of sequence input.
        if (random.random() < self.sequenceProbability):
            outputGrid = self.inputs[self.patIndex][self.index]
        else:
            sequenceLen = len(self.inputs[self.patIndex])
            outputGrid = self.inputs[self.patIndex][random.randint(0, sequenceLen-1)]
        # Increment the index for next time
        self.index += 1
        if (self.index >= len(self.inputs[self.patIndex])):
            self.index = 0
        # If noise was added return the noisy grid.
        if newGrid is not None:
            return newGrid
        else:
            return outputGrid

