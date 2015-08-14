import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math


'''
A class used to calculate the overlap values for columns
in a single HTM layer. This class uses theano functions
to speed up the computation. Can be implemented on a GPU
see theano documents for enabling GPU calculations.

Take a numpy input 2D matrix and convert this into a
theano tensor where the tensor holds inputs that are connected
by potential synapses to columns.

Eg
input = [[0,0,1,0]
         [1,0,0,0]
         [0,1,0,1]
         [0,0,0,0]
         [0,1,0,0]]

Output = [[x1,x2,x3,x4]
          [x5,x6,x7,x8]
          [x9,x10,x11,x12]
          [x13,x14,x15,x16]
          [x17,x18,x19,x20]]

x10 = [1,0,0,0,1,0,0,0,0]

potential_width = 3
potential_height = 3

Take this output and calcuate the overlap for each column.
This is the sum of 1's for each columns input.

'''


class OverlapCalculator():
    def __init__(self, potentialWidth, potentialHeight,
                 stepX, stepY,
                 inputWidth, inputHeight,
                 centerPotSynapses, connectedPerm,
                 minOverlap):
        # Overlap Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerPotSynapses = centerPotSynapses
        self.potentialWidth = potentialWidth
        self.potentialHeight = potentialHeight
        self.connectedPermParam = connectedPerm
        self.minOverlap = minOverlap
        # StepX and Step Y describe how far each
        # columns potential synapses differ from the adjacent
        # columns in the X and Y directions.
        self.stepX = stepX
        self.stepY = stepY
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        # Calculate how many columns are expected from these
        # parameters.
        numColumnsWidth = int(math.ceil(float(self.inputWidth)/float(self.stepX)))
        numColumnsHeight = int(math.ceil(float(self.inputHeight)/float(self.stepY)))
        self.numColumns = numColumnsWidth * numColumnsHeight

        # Check that these input parameters are ok
        self.assertInputs()

        # Create theano variables and functions
        ############################################

        # Create the theano function for calculating
        # the inputs to a column from an input grid.
        self.kernalSize = (potentialHeight, potentialWidth)
        # poolstep is how far to move the kernal in each direction.
        self.poolstep = (self.stepY, self.stepX)
        # Create the theano function for calculating the input to each column
        self.neib_shape = T.as_tensor_variable(self.kernalSize)
        self.neib_step = T.as_tensor_variable(self.poolstep)
        self.pool_inp = T.tensor4('pool_input', dtype='float32')
        self.pool_convole = images2neibs(self.pool_inp, self.neib_shape, self.neib_step, mode='valid')
        self.pool_inputs = function([self.pool_inp],
                                    [self.pool_convole],
                                    on_unused_input='warn',
                                    allow_input_downcast=True)

        # Create the theano function for calculating
        # which synapses are connected.
        self.j = T.matrix('poolConnInput', dtype='float32')
        self.k = T.matrix('synInputVal', dtype='float32')
        self.connectedPermanence = T.matrix('con_perm', dtype='float32')
        # Compare the input matrix j to the scalar parameter.
        # If the matrix value is less then the connectedPermParam
        # return zero.
        self.checkConn = T.switch(T.lt(self.connectedPermParam, self.j), self.k, 0.0)
        # Use enable downcast so the numpy arrays of float 64 can be downcast to float32
        self.getConnectedSynInput = function([self.j, self.k],
                                             self.checkConn,
                                             mode=Mode(linker='vm'),
                                             allow_input_downcast=True)

        # Create the theano function for calculating
        # the overlap of each col
        self.b = T.matrix(dtype='float32')
        self.m = self.b.sum(axis=1)
        self.calcOverlap = function([self.b], self.m, allow_input_downcast=True)

        # Create the theano function for calculating
        # if an overlap value is larger then minOverlap.
        # If not then set to zero.
        self.currOverlap = T.vector(dtype='float32')
        self.ch_over = T.switch(T.lt(self.minOverlap, self.currOverlap), self.currOverlap, 0.0)
        self.checkMinOverlap = function([self.currOverlap],
                                        self.ch_over,
                                        allow_input_downcast=True)

    def assertInputs(self):
        # Make sure the input parameters are valid values.
        assert self.stepX > 0
        assert self.stepY > 0
        assert self.stepX < self.inputWidth
        assert self.stepY < self.inputHeight

    def checkNewInputParams(self, newColSynPerm, newInput):
        # Check that the new input has the same dimensions as the
        # originally defined input parameters.
        assert self.inputWidth == len(newInput[0])
        assert self.inputHeight == len(newInput)
        assert self.potentialWidth * self.potentialHeight == len(newColSynPerm[0])
        # Check the number of rows in the newColSynPerm matrix equals
        # the number of expected columns.
        assert self.numColumns == len(newColSynPerm)

    def addPaddingToInput(self, inputGrid):
        topPos_y = 0
        bottomPos_y = 0
        leftPos_x = 0
        rightPos_x = 0

        if self.centerPotSynapses == 0:
            # The potential synapses are not centered over the input
            # This means only the right side and bottom of the input
            # need padding.
            topPos_y = 0
            bottomPos_y = self.potentialHeight-1
            leftPos_x = 0
            rightPos_x = self.potentialWidth-1
        else:
            # The potential synapses are centered over the input
            # This means all sides of the input may need padding
            topPos_y = self.potentialHeight/2
            if ((self.inputHeight-1) % self.stepY == 0):
                bottomPos_y = int(math.ceil(self.potentialHeight/2.0))-1
            else:
                botLeftOver = (self.inputHeight-1) % self.stepY
                halfPotHeight = int(math.ceil(self.potentialHeight/2.0))-1
                if (halfPotHeight - botLeftOver) > 0:
                    bottomPos_y = halfPotHeight - botLeftOver

            leftPos_x = self.potentialWidth/2
            if ((self.inputWidth-1) % self.stepX == 0):
                rightPos_x = int(math.ceil(self.potentialWidth/2.0))-1
            else:
                rightLeftOver = (self.inputWidth-1) % self.stepX
                halfPotWidth = int(math.ceil(self.potentialWidth/2.0))-1
                if (halfPotWidth - rightLeftOver) > 0:
                    rightPos_x = halfPotWidth - rightLeftOver

        # Make sure all are larger then zero still
        if topPos_y < 0:
            topPos_y = 0
        if bottomPos_y < 0:
            bottomPos_y = 0
        if leftPos_x < 0:
            leftPos_x = 0
        if rightPos_x < 0:
            rightPos_x = 0

        # Add the padding around the edges of the inputGrid
        inputGrid = np.lib.pad(inputGrid,
                              ((0, 0),
                               (0, 0),
                               (topPos_y, bottomPos_y),
                               (leftPos_x, rightPos_x)),
                               'constant',
                               constant_values=(0))

        print "inputGrid = \n%s" % inputGrid

        return inputGrid

    def getColInputs(self, inputGrid):
        # Take the input and put it into a 4D tensor.
        # This is because the theano function images2neibs
        # works with 4D tensors only.
        inputGrid = np.array([[inputGrid]])

        #print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # work out how much padding is needed on the borders
        # using the defined potential width and potential height.
        inputGrid = self.addPaddingToInput(inputGrid)

        print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        print "self.potentialWidth = %s" % self.potentialWidth
        print "self.potentialHeight = %s" % self.potentialHeight
        print "self.stepX = %s, self.stepY = %s" % (self.stepX, self.stepY)
        # Calculate the inputs to each column.
        inputConPotSyn = self.pool_inputs(inputGrid)
        # The returned array is within a list so just use pos 0.
        #print "inputConPotSyn = \n%s" % inputConPotSyn[0]
        print "inputConPotSyn.shape = %s,%s" % inputConPotSyn[0].shape
        return inputConPotSyn[0]

    def calculateOverlap(self, colSynPerm, inputGrid):
        # Check that the new inputs are the same dimensions as the old ones
        # and the colsynPerm match the original specified parameters.
        self.checkNewInputParams(colSynPerm, inputGrid)
        # Calcualte the inputs to each column
        colInputPotSyn = self.getColInputs(inputGrid)
        # Call the theano functions to calculate the overlap value.
        print "len(colSynPerm) = %s len(colSynPerm[0]) = %s " % (len(colSynPerm), len(colSynPerm[0]))
        print "len(colInputPotSyn) = %s len(colInputPotSyn[0]) = %s " % (len(colInputPotSyn), len(colInputPotSyn[0]))
        connectedSynInputs = self.getConnectedSynInput(colSynPerm, colInputPotSyn)
        #print "connectedSynInputs = \n%s" % connectedSynInputs
        colOverlapVals = self.calcOverlap(connectedSynInputs)
        print colOverlapVals
        return colOverlapVals, colInputPotSyn

    def removeSmallOverlaps(self, colOverlapVals):
        # Set any overlap values that are smaller then the
        # minOverlap value to zero.
        self.minOverlap = minOverlap
        newColOverlapVals = self.checkMinOverlap(colOverlapVals)
        #print "newColOverlapVals \n%s" % newColOverlapVals
        return newColOverlapVals


if __name__ == '__main__':

    potWidth = 4
    potHeight = 1
    centerPotSynapses = 1
    numInputRows = 8
    numInputCols = 10
    numColumnRows = 8
    numColumnCols = 4
    connectedPerm = 0.3
    minOverlap = 3
    numPotSyn = potWidth * potHeight
    numColumns = numColumnRows * numColumnCols

    # Calculate what the stepX and stepY values should be so
    # that the entire input is covered by the potential Synapses.
    stepX = 4
    stepY = 1

    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # To get the above array from a htm use
    # allCols = self.htm.regionArray[0].layerArray[0].columns.flatten()
    # colPotSynPerm = np.array([[allCols[j].potentialSynapses[i].permanence for i in range(36)] for j in range(1600)])

    print "colSynPerm = \n%s" % colSynPerm
    newInputMat = np.random.randint(2, size=(numInputRows, numInputCols))

    # Create an instance of the overlap calculation class
    overlapCalc = OverlapCalculator(potWidth,
                                    potHeight,
                                    stepX,
                                    stepY,
                                    numInputCols,
                                    numInputRows,
                                    centerPotSynapses,
                                    connectedPerm,
                                    minOverlap)

    print "newInputMat = \n%s" % newInputMat
    #potSyn = np.random.rand(1, 1, 4, 4)

    # Return both the overlap values and the inputs from
    # the potential synapses to all columns.
    colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)
    print "len(colOverlaps) = %s" % len(colOverlaps)
    print "colOverlaps = \n%s" % colOverlaps

    # limit the overlap values so they are larger then minOverlap
    colInputs = overlapCalc.removeSmallOverlaps(colOverlaps)

    print "colPotInputs = \n%s" % colPotInputs


