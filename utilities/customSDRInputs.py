import numpy as np
from copy import deepcopy
import random
from utilities import sdrFunctions as SDRFunct


class customSDRInputs:
    '''
    A class used to store different input sequences.
    A number of different sequences are stored in
    an array of an array of matricies.

    New sequences are added by calling the addSequence function.
    This stores the sequence in it's own array within the list of sequences.

    New sequeces can have different numbers of patterns but all the patterns
    must have the same dimensions. This is because the htm can't handle inputs
    that changed sizes. This is a limitation of the current htm design.


    '''
    def __init__(self, width, height, numInputs):
        # The number of inputs to store
        self.numInputPatterns = 0
        # An index indicating the current pattern that is being used as a serias of input grids.
        self.patIndex = 0
        self.width = width
        self.height = height
        # A list storing different input patterns
        # Each pattern is a series of 2dArray grids storing binary SDR patterns.
        self.inputs = []
        self._createDefaultSequence(self.width, self.height, numInputs)
        # Use an index to keep track of which input to send next.
        self.index = 0
        # A variable speifying the amount of noise in the inputs 0 to 1
        self.noise = 0.0
        # A variable indicating the chance that the next input is a random input from the sequence.
        # This variable is used to create an input sequence that sometimes changes. It is the probablity
        # that the next input is the correct input in the sequence
        self.sequenceProbability = 1.0

    def _createDefaultSequence(self, width, height, numInputs):
        # Create the input patterns to test with

        newPattern = np.array([[[0 for i in range(width)]
                                for j in range(height)]
                              for k in range(numInputs)])
        # The first pattern is just a vertical line of width 1.
        # It moves left to right.
        for n in range(len(newPattern)):
            for y in range(len(newPattern[0])):
                for x in range(len(newPattern[n][y])):
                    if x == n:
                        newPattern[n][y][x] = 1
        # Store the first pattern
        self.inputs.append(newPattern)

    def getNumCustomSequences(self):
        # Return the number of user defined seuences.
        # This is just the length of the self.inputs list minus one
        # since the first sequence is the deafualt one.
        return (len(self.inputs)-1)

    def appendSequence(self, sequence):
        # Take the input sequence and append it to the end of the current
        # inputs list which holds a list of all the other sequences.
        # Return the position index that this new sequence hold in the inputs list
        # of sequences.
        # Check that the new sequecnee has the same size as the current
        # stored sequences.
        assert len(sequence[0]) == self.height
        assert len(sequence[0][0]) == self.width
        self.inputs.append(sequence)
        self.numInputPatterns += 1
        return self.numInputPatterns

    def getNumInputsInSeq(self, patIndex):
        # Return the number of patterns in a particular sequence.
        return len(self.inputs[patIndex])

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
        # Give the next output a chance to be an out of sequence input.
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

