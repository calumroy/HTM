import numpy as np
from copy import deepcopy
import random
import sdrFunctions as SDRFunct
import glob
import os
from scipy import misc
import PIL


class imageInputs:
    '''
    A class used to create different input sequences consisting of
    binary images. A number of different sequences are created by combining the
    different binary images into sequences.
    These sequences are stored as a list of an array of matricies.

    The different sequences are defined in the setinputs method.
    To add a new sequence, create a new directory for each new sequence.
    The images should be stored as number indicating there order in the sequence.

    '''
    def __init__(self, test_seq_dir):
        # The number of inputs for each input sequence of patterns.
        self.numInputs = None
        self.width = None
        self.height = None
        # How many input input sequence patterns to store
        self.numPatterns = 8
        # An index indicating the current pattern that is being used as a series of input grids.
        self.patIndex = 0
        self.inputs = []
        # An array storing different input patterns
        # Each pattern is a series of 2dArray grids storing binary patterns.
        # self.inputs = np.array([[[[0 for i in range(self.width)]
        #                        for j in range(self.height)]
        #                        for k in range(self.numInputs)]
        #                        for l in range(self.numPatterns)])
        # Set the self.inputs array to a sequence of np.arrays
        # containing the sequence images from the image test directory.
        self.setInputs(test_seq_dir)
        # Use an index to keep track of which input to send next
        self.index = 0
        # A variable specifying the amount of noise in the inputs 0 to 1
        self.noise = 0.0
        # A variable indicating the chance that the next input is a random input from the sequence.
        # This variable is used to create an input sequence that sometimes changes. It is the probablity
        # that the next input is the correct input in the sequence
        self.sequenceProbability = 1.0

    def setInputs(self, test_seq_dir):
        # Import the sequences stored in the sequence directory
        self.inputs = self.importAllInputSequences(test_seq_dir)

    def importAllInputSequences(self, directory):

        base_dir = "./"
        png = []
        pic_postfix = ".png"
        test_seq_dir = None
        # Find the directory containing sub dirs with sequences of images.
        print "searching in root directory ", base_dir
        for root, dirs, filesnames in os.walk(base_dir):
            for dir_name in dirs:
                # print 'searching', dir_name
                if directory in dir_name:
                    print 'found dir', dir_name
                    test_seq_dir = os.path.join(root, directory)
                    break

        # Now find all the sub dirs. Each sub dir is another sequence of images.
        for root, dirs, filesnames in os.walk(test_seq_dir):
            for dir_name in dirs:
                # print "     searching", dir_name
                # print os.path.join(root,directory)
                png.append([])
                pic_search_str = os.path.join(root, dir_name)+"/*"+pic_postfix
                for file_name in glob.glob(pic_search_str):
                    # print "         looking at file ", file_name
                    # help(misc.imread)
                    png[-1].append(misc.imread(file_name, flatten=1))

        im = np.asarray(png)
        im_bin = (im < 100).astype(int)

        print 'Importing image sequences: done Shape:', im_bin.shape
        # print "first image in first sequence = \n"
        # print im_bin[0][0]
        return im_bin

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
            assert (len(self.inputs) > 0)
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


# if __name__ == '__main__':
#     InputCreator = imageInputs(r'test_seqs')