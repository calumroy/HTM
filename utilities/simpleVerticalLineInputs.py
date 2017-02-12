import numpy as np
from copy import deepcopy
import random
import sdrFunctions as SDRFunct


class simpleVerticalLineInputs:
    '''
    A class used to create different input sequences consisting of
    straight vertical lines. A number of different sequences are stored in
    an array of an array of matricies.

    '''
    def __init__(self, width, height, numInputs):
        # The number of inputs to store for each pattern
        self.numInputs = numInputs
        self.width = width
        self.height = height
        # An index indicating the current pattern that is being used as a series of input grids.
        self.patIndex = 0
        # An array storing different input patterns
        # Each pattern is a series of 2dArray grids storing binary patterns.
        # self.inputs = [np.array([[[0 for i in range(self.width)]
        #                         for j in range(self.height)]
        #                         for k in range(self.numInputs)])
        #                         for l in range(self.numPatterns)]
        self.inputs = []

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
        numInputs = self.numInputs
        new_inputs0 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs0)):
            for y in range(len(new_inputs0[0])):
                for x in range(len(new_inputs0[n][y])):
                    if x == n:
                        new_inputs0[n][y][x] = 1
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs0)
        

        # The second input pattern moves right to left
        numInputs = self.numInputs
        new_inputs1 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs1)):
            for y in range(len(new_inputs1[0])):
                for x in range(len(new_inputs1[n][y])):
                    # reverse the pattern
                    if x == (len(new_inputs1) - 1 - n):
                        new_inputs1[n][y][x] = 1
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs1)

        # The third pattern is just every second input of the first pattern
        patIndex = 0
        numInputs = self.numInputs
        new_inputs2 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs2)):
            patIndex = patIndex + 4
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs2[n] = inputs[0][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs2)


        # The forth pattern is just every second input of the second pattern
        patIndex = 0
        numInputs = self.numInputs
        new_inputs3 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs3)):
            patIndex = patIndex + 4
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs3[n] = inputs[1][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs3)

        # The fifth pattern is the third pattern then the forth pattern
        patIndex = 0
        numInputs = self.numInputs
        new_inputs4 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs4)):
            patIndex = patIndex + 1
            if patIndex >= self.numInputs:
                patIndex = 0
            if patIndex <= int(self.numInputs/2):
                new_inputs4[n] = inputs[2][patIndex]
            else:
                new_inputs4[n] = inputs[3][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs4)

        # The 6th pattern is the first combined with the third pattern
        # by a logical or operation.
        patIndex = 0
        numInputs = self.numInputs
        new_inputs5 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(inputs[2])):
            patIndex = patIndex + 1
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs5[n] = SDRFunct.orSDRPatterns(inputs[0][patIndex], inputs[2][patIndex])
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs5)

        # The seventh pattern is just every 6th input of the first pattern
        patIndex = 0
        numInputs = self.numInputs
        new_inputs6 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs6)):
            patIndex = patIndex + 6
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs6[n] = inputs[0][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs6)

        # The eighth pattern is just every 6th input of the second pattern
        patIndex = 0
        numInputs = self.numInputs
        new_inputs7 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(len(new_inputs7)):
            patIndex = patIndex + 6
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs7[n] = inputs[1][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs7)

        # The ninth pattern is just every second input of the first pattern
        # but for only a few of the first inputs.
        patIndex = 0
        numInputs = int(np.floor(self.numInputs/7.0))
        new_inputs8 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(numInputs):
            patIndex = patIndex + 2
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs8[n] = inputs[0][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs8)    

        # The tenth pattern is just every second input of the first pattern
        # but for only a few of the first inputs starting at a different input.
        patIndex = 1
        numInputs = int(np.floor(self.numInputs/7.0))
        new_inputs9 = np.array([[[0 for i in range(self.width)]
                                 for j in range(self.height)]
                                 for k in range(numInputs)])
        for n in range(numInputs):
            patIndex = patIndex + 2
            if patIndex >= self.numInputs:
                patIndex = 0
            new_inputs9[n] = inputs[0][patIndex]
        # Add the new pattern to the classes list storing all the patterns.
        inputs.append(new_inputs9) 

    def changePattern(self, patternIndex):
        # Change the input pattern
        self.patIndex = patternIndex

    def getNumInputsPat(self, patternIndex):
        numInputsPatX = len(self.inputs[patternIndex])
        return numInputsPatX

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
        if (random.random() <= self.sequenceProbability):
            outputGrid = self.inputs[self.patIndex][self.index]
        else:
            sequenceLen = len(self.inputs[self.patIndex])
            outputGrid = self.inputs[self.patIndex][random.randint(0, sequenceLen-1)]
        # Increment the index for next time
        self.index += 1
        if (self.index >= len(self.inputs[self.patIndex])):
            self.index = 0

        #import ipdb; ipdb.set_trace()
        # If noise was added return the noisy grid.
        if newGrid is not None:
            return newGrid
        else:
            return outputGrid

