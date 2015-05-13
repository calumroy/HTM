import numpy as np
from copy import deepcopy
import random

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

    def getReward(self):
        # Required function for a InputCreator class
        reward = 0
        return reward

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

    def orSDRPatterns(self, SDR1, SDR2):
        '''
        Combine two inputs SDR patterns to create an output SDR
        which is the or of both the input SDRs.
        '''

        outputSDR = None

        if SDR1 is not None:
            outputSDR = np.logical_or(SDR1, SDR2).astype(int)

        return outputSDR

