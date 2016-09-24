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
                 columnsWidth, columnsHeight,
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
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        # Calculate how many columns are expected from these
        # parameters.
        self.columnsWidth = columnsWidth
        self.columnsHeight = columnsHeight
        self.numColumns = columnsWidth * columnsHeight
        # Store the potetnial inputs to every column.
        # Each row represents the inputs a columns potential synapses cover.
        self.colInputPotSyn = None
        # Store the potential overlap values for every column
        self.colPotOverlaps = None
        # StepX and Step Y describe how far each
        # columns potential synapses differ from the adjacent
        # columns in the X and Y directions. These parameters can't
        # change as theano uses them to setup functions.
        self.stepX, self.stepY = self.getStepSizes(inputWidth, inputHeight,
                                                   self.columnsWidth, self.columnsHeight,
                                                   self.potentialWidth, self.potentialHeight)
        # Contruct a tiebreaker matrix for the columns potential synapses.
        # It contains small values that help resolve any ties in potential
        # overlap scores for columns.
        self.potSynTieBreaker = np.array([[0.0 for i in range(self.potentialHeight*self.potentialWidth)]
                                         for j in range(self.numColumns)])
        self.makePotSynTieBreaker(self.potSynTieBreaker)
        # Store the potential inputs to every column plus the tie breaker value.
        # Each row represents the inputs a columns potential synapses cover.
        self.colInputPotSynTie = np.array([[0.0 for i in range(self.potentialHeight*self.potentialWidth)]
                                          for j in range(self.numColumns)])
        self.colTieBreaker = np.array([0.0 for i in range(self.numColumns)])
        self.makeColTieBreaker(self.colTieBreaker)

        # Create theano variables and functions
        ############################################

        # Create the theano function for calculating
        # the multiplication elementwise of 2 matricies.
        self.i_grid = T.matrix(dtype='float32')
        self.j_grid = T.matrix(dtype='float32')
        self.multi_vals = self.i_grid * self.j_grid
        self.multi_grids = function([self.i_grid, self.j_grid],
                                    self.multi_vals,
                                    on_unused_input='warn',
                                    allow_input_downcast=True)

        # Create the theano function for calculating
        # the addition of a small tie breaker value to each matrix input.
        self.o_grid = T.matrix(dtype='float32')
        self.tie_grid = T.matrix(dtype='float32')
        self.add_vals = T.add(self.o_grid, self.tie_grid)
        self.add_tieBreaker = function([self.o_grid, self.tie_grid],
                                       self.add_vals,
                                       on_unused_input='warn',
                                       allow_input_downcast=True)

        # Create the theano function for calculating
        # the addition of a small tie breaker value to each matrix input.
        self.o_vect = T.vector(dtype='float32')
        self.tie_vect = T.vector(dtype='float32')
        self.add_vectVals = T.add(self.o_vect, self.tie_vect)
        self.add_vectTieBreaker = function([self.o_vect, self.tie_vect],
                                           self.add_vectVals,
                                           on_unused_input='warn',
                                           allow_input_downcast=True)

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
                                    self.pool_convole,
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
        # if an overlap value is greater then minOverlap.
        # If not then set to zero.
        self.currOverlap = T.vector(dtype='float32')
        self.ch_over = T.switch(T.ge(self.currOverlap, self.minOverlap), self.currOverlap, 0.0)
        self.checkMinOverlap = function([self.currOverlap],
                                        self.ch_over,
                                        allow_input_downcast=True)

        # Create the theano function for calculating
        # the x and y indicies from a input element index.
        # The input matrix contains a number representing a potential
        # synapse and the position that the synpase connects to in the input
        # grid. Convert this into a col, row index and output it into 2 matricies,
        # one for the row number the second for the columns number.
        # This gives the position info for all the potential synapse for every column.
        self.inputGridWidth = T.scalar(dtype='int32')
        #self.inputGridHeight = T.scalar(dtype='int32')
        self.inputInd = T.matrix(dtype='int32')
        self.potSyn_XYInd = (self.inputInd / self.inputGridWidth,
                             self.inputInd % self.inputGridWidth)
        self.check_notpadding = T.switch(T.gt(self.inputInd, 0), self.potSyn_XYInd, -1)
        self.convert_indicesToXY = function([self.inputGridWidth,
                                            #self.inputGridHeight,
                                            self.inputInd],
                                            self.potSyn_XYInd,
                                            allow_input_downcast=True)

        ########################### END THEANO ###############################

    def makePotSynTieBreaker(self, tieBreaker):
        # create a tie breaker matrix holding small values for each element in
        # the self.colInputPotSyn grid. The tie breaker values are created such that
        # for a particular row in the colInputPotSyn adding all tie breaker values up
        # the result is less then 1. We will make it less then 0.5. THe tie breaker values
        # are all multiples of the same number. Each row in the colInputPotSyn grid
        # has a different pattern of tie breaker values. THis is done by sliding the previous
        # rows values along by 1 and wrapping at the end of the row.
        # They are used  to resolve situations where columns have the same overlap number.
        # The purpose of
        inputHeight = len(tieBreaker)
        inputWidth = len(tieBreaker[0])
        #numInputs = inputWidth

        # Use the sum of all integer values less then or equal to formula.
        # This is because each row has its tie breaker values added together.
        # We want to make sure the result from adding the tie breaker values is
        # less then 0.5 but more then 0.0.
        n = float(inputWidth)
        normValue = float(0.5/(n*(n+1.0)/2.0))
        #normValue = 1.0/float(2*inputWidth+2)
        print "maxNormValue = %s" % (n*(n+1.0)/2.0)
        print "normValue = %s" % normValue
        print "tie Width = %s" % inputWidth

        rowsTie = np.arange(inputWidth)+1
        rowsTie = rowsTie*normValue
        # Create a tiebreaker that changes for each row.
        for j in range(len(tieBreaker)):
            tieBreaker[j] = np.roll(rowsTie, j)
            #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

            #print "np.roll(rowsTie, j) * inputWidth = %s" % (np.roll(rowsTie, j) * inputWidth)

            #for i in range(len(tieBreaker[0])):

                # if (j % 2) == 1:
                #     # For odd positions bias to the bottom left
                #     tieBreaker[j][i] = ((j+1)*inputWidth+(inputWidth-i-1))*normValue
                # else:
                #     # For even positions bias to the bottom right
                #     tieBreaker[j][i] = (1+i+j*inputWidth)*normValue
        # print "self.tieBreaker = \n%s" % self.tieBreaker

    def makeColTieBreaker(self, tieBreaker):
        # Make a vector of tiebreaker values to add to the columns overlap values vector.
        normValue = 1.0/float(2*self.numColumns+2)

        # Create a tiebreaker that is not biased to either side of the columns grid.
        for j in range(len(tieBreaker)):
            # The tieBreaker is a flattened vecto of the columns overlaps.
            # workout the row and col number of the non flattened matrix.
            rowNum = math.floor(j/self.columnsWidth)
            colNum = j % self.columnsWidth
            if (j % 2) == 1:
                # For odd positions bias to the bottom left
                tieBreaker[j] = ((rowNum+1)*self.columnsWidth+(self.columnsWidth-colNum-1))*normValue
            else:
                # For even positions bias to the bottom right
                tieBreaker[j] = (1+colNum+rowNum*self.columnsWidth)*normValue

    def checkNewInputParams(self, newColSynPerm, newInput):
        # Check that the new input has the same dimensions as the
        # originally defined input parameters.
        assert self.inputWidth == len(newInput[0])
        assert self.inputHeight == len(newInput)
        assert self.potentialWidth * self.potentialHeight == len(newColSynPerm[0])
        # Check the number of rows in the newColSynPerm matrix equals
        # the number of expected columns.
        assert self.numColumns == len(newColSynPerm)

    def addPaddingToInput(self, inputGrid, useZeroPadVal=True):
        # Add padding elements to the input Grid so that the
        # convole function images2neibs can convole over the input.
        topPos_y = 0
        bottomPos_y = 0
        leftPos_x = 0
        rightPos_x = 0

        # This calcualtes how much of the input grid is not covered by
        # the htm grid in each dimension using the step sizes.
        leftOverWidth = self.inputWidth - (1 + (self.columnsWidth - 1) * self.stepX)
        leftOverHeight = self.inputHeight - (1 + (self.columnsHeight - 1) * self.stepY)

        if self.centerPotSynapses == 0:
            # The potential synapses are not centered over the input
            # This means only the right side and bottom of the input
            # need padding.
            topPos_y = 0
            bottomPos_y = int(math.floor(self.potentialHeight-1) - math.floor(leftOverHeight))
            leftPos_x = 0
            rightPos_x = int(math.floor(self.potentialWidth-1) - math.floor(leftOverWidth))

        else:
            # The potential synapses are centered over the input
            # This means all sides of the input may need padding
            topPos_y = int(math.ceil(float(self.potentialHeight-1)/2) - math.ceil(float(leftOverHeight)/2))
            bottomPos_y = int(math.floor(float(self.potentialHeight-1)/2) - math.floor(float(leftOverHeight)/2))

            leftPos_x = int(math.ceil(float(self.potentialWidth-1)/2) - math.ceil(float(leftOverWidth)/2))
            rightPos_x = int(math.floor(float(self.potentialWidth-1)/2) - math.floor(float(leftOverWidth)/2))

        # Make sure all are larger then zero still
        if topPos_y < 0:
            topPos_y = 0
        if bottomPos_y < 0:
            bottomPos_y = 0
        if leftPos_x < 0:
            leftPos_x = 0
        if rightPos_x < 0:
            rightPos_x = 0

        # Padding value
        if useZeroPadVal is False:
            padValue = -1
        else:
            padValue = 0

        # Add the padding around the edges of the inputGrid
        inputGrid = np.lib.pad(inputGrid,
                               ((0, 0),
                                (0, 0),
                                (topPos_y, bottomPos_y),
                                (leftPos_x, rightPos_x)),
                               'constant',
                               constant_values=(padValue))

        # print "inputGrid = \n%s" % inputGrid

        return inputGrid

    def getPotentialSynapsePos(self, inputWidth, inputHeight):
        # Return 2 matricies of x and y positions repectively
        # that each column potential synpases connects to in the input.
        # First create a matrix with the same dimensions as
        # the inputGrid but where each position holds an index.
        # The index is just a number representing that element in the inputGrid.
        indexInputGrid = np.array([[i + j*inputWidth for i in range(inputWidth)] for j in range(inputHeight)])
        # print "indexInputGrid = \n%s" % indexInputGrid
        # Take the input and put it into a 4D tensor.
        # This is because the theano function images2neibs
        # works with 4D tensors only.
        indexInputGrid = np.array([[indexInputGrid]])

        # Work out how far each columns pool of inputs should step
        # so the entire input is covered equally in the convole.
        self.stepX, self.stepY = self.getStepSizes(inputWidth, inputHeight,
                                                   self.columnsWidth, self.columnsHeight,
                                                   self.potentialWidth, self.potentialHeight)
        # print "self.stepX = %s, self.stepY = %s" % (self.stepX, self.stepY)
        # work out how much padding is needed on the borders
        # using the defined potential width and potential height.
        indexInputGrid = self.addPaddingToInput(indexInputGrid, False)

        # print "padded InputGrid = \n%s" % indexInputGrid
        # print "padded InputGrid.shape = %s,%s,%s,%s" % indexInputGrid.shape
        # print "self.potentialWidth = %s" % self.potentialWidth
        # print "self.potentialHeight = %s" % self.potentialHeight
        # print "self.stepX = %s, self.stepY = %s" % (self.stepX, self.stepY)
        # print "inputWidth = %s, inputHeight = %s" % (inputWidth, inputHeight)
        # Calculate the inputs to each column.
        inputPotSynIndex = self.pool_inputs(indexInputGrid)
        # print "inputPotSynIndex = \n%s" % inputPotSynIndex

        # Now turn the inputPotSynIndex into two matricies where the first
        # holds the x and the second holds the y indicies for the element
        # in the inpuGrid that a potential synapse connects to.
        potSynXYIndex = self.convert_indicesToXY(inputWidth,
                                                 inputPotSynIndex)
        # print "potSynXYIndex = \n%s" % potSynXYIndex

        return potSynXYIndex

    def getStepSizes(self, inputWidth, inputHeight, colWidth, colHeight, potWidth, potHeight):
        # Work out how large to make the step sizes so all of the
        # inputGrid can be covered as best as possible by the columns
        # potential synapses.

        stepX = int(round(float(inputWidth)/float(colWidth)))
        stepY = int(round(float(inputHeight)/float(colHeight)))

        #import ipdb; ipdb.set_trace()
        # The step sizes may need to be increased if the potential sizes are too small.
        if potWidth + (colWidth-1)*stepX < inputWidth:
            # Calculate how many of the input elements cannot be covered with the current stepX value.
            uncoveredX = (inputWidth - (potWidth + (colWidth - 1) * stepX))
            # Use this to update the stepX value so all input elements are covered.
            stepX = stepX + int(math.ceil(float(uncoveredX) / float(colWidth-1)))

        if potHeight + (colHeight-1)*stepY < self.inputHeight:
            uncoveredY = (inputHeight - (potHeight + (colHeight - 1) * stepY))
            stepY = stepY + int(math.ceil(float(uncoveredY) / float(colHeight-1)))

        return stepX, stepY

    def addVectTieBreaker(self, vectorVals, tieBreaker):
        # Add a tieBreaker array to the vector array.
        gridPlusTieB = self.add_vectTieBreaker(vectorVals, tieBreaker)
        return gridPlusTieB

    def maskTieBreaker(self, grid, tieBreaker):
        # Multiply the tiebreaker values by the input grid then add them to it.
        # Since the grid contains ones and zeros some tiebreaker values are
        # masked out. This means the tie breaker will be different for each input
        # pattern.
        maskedTieBreaker = None

        maskedTieBreaker = self.multi_grids(grid, tieBreaker)
        print "maskedTieBreaker = \n%s" % maskedTieBreaker

        gridPlusTieB = self.add_tieBreaker(grid, maskedTieBreaker)
        return gridPlusTieB

    def getColInputs(self, inputGrid):
        # This function uses theano's convolution function to
        # return the inputs that each column potentially connects to.

        # Take the input and put it into a 4D tensor.
        # This is because the theano function images2neibs
        # works with 4D tensors only.
        inputGrid = np.array([[inputGrid]])

        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # work out how much padding is needed on the borders
        # using the defined potential width and potential height.
        inputGrid = self.addPaddingToInput(inputGrid)

        # print "padded InputGrid = \n%s" % inputGrid
        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        # print "self.potentialWidth = %s" % self.potentialWidth
        # print "self.potentialHeight = %s" % self.potentialHeight
        # print "self.stepX = %s, self.stepY = %s" % (self.stepX, self.stepY)
        # Calculate the inputs to each column.
        inputConPotSyn = self.pool_inputs(inputGrid)
        # The returned array is within a list so just use pos 0.
        # print "inputConPotSyn = \n%s" % inputConPotSyn
        # print "inputConPotSyn.shape = %s,%s" % inputConPotSyn.shape
        return inputConPotSyn

    def getPotentialOverlaps(self):
        # Return the calculated potential overlap score for every column.
        # This is the overlap score each column has if all poential synpases
        # are checked for active inputs.
        return self.colPotOverlaps

    def calculateOverlap(self, colSynPerm, inputGrid):
        # Check that the new inputs are the same dimensions as the old ones
        # and the colsynPerm match the original specified parameters.
        self.checkNewInputParams(colSynPerm, inputGrid)
        # Calcualte the inputs to each column
        self.colInputPotSyn = self.getColInputs(inputGrid)

        # Add a masked small tiebreaker value to the self.colInputPotSyn scores.
        self.colInputPotSynTie = self.maskTieBreaker(self.colInputPotSyn, self.potSynTieBreaker)
        #print "self.colInputPotSyn = \n%s" % self.colInputPotSyn
        #print "self.colInputPotSynTie = \n%s" % self.colInputPotSynTie

        # Calculate the potential overlap scores for every column.
        # Sum the potential inputs for every column.
        self.colPotOverlaps = self.calcOverlap(self.colInputPotSynTie)
        #print "self.colPotOverlaps = \n%s" % self.colPotOverlaps

        # Call the theano functions to calculate the overlap value.
        #print "colSynPerm = \n%s" % colSynPerm
        # print "colInputPotSyn = \n%s" % colInputPotSyn
        # print "len(colSynPerm) = %s len(colSynPerm[0]) = %s " % (len(colSynPerm), len(colSynPerm[0]))
        # print "len(colInputPotSyn) = %s len(colInputPotSyn[0]) = %s " % (len(colInputPotSyn), len(colInputPotSyn[0]))
        connectedSynInputs = self.getConnectedSynInput(colSynPerm, self.colInputPotSyn)
        #print "connectedSynInputs = \n%s" % connectedSynInputs
        colOverlapVals = self.calcOverlap(connectedSynInputs)

        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        #print "self.colTieBreaker = \n%s" % self.colTieBreaker.reshape((self.columnsWidth, self.columnsHeight))
        # Add a small tiebreaker value to the column overlap scores vector.
        colOverlapVals = self.addVectTieBreaker(colOverlapVals, self.colTieBreaker)

        #print "colOverlapVals = \n%s" % colOverlapVals.reshape((self.columnsWidth, self.columnsHeight))
        return colOverlapVals, self.colInputPotSyn

    def removeSmallOverlaps(self, colOverlapVals):
        # Set any overlap values that are smaller then the
        # minOverlap value to zero.
        newColOverlapVals = self.checkMinOverlap(colOverlapVals)
        # print "newColOverlapVals \n%s" % newColOverlapVals
        return newColOverlapVals


if __name__ == '__main__':

    potWidth = 4
    potHeight = 4
    centerPotSynapses = 1
    numInputRows = 4
    numInputCols = 5
    numColumnRows = 7
    numColumnCols = 5
    connectedPerm = 0.3
    minOverlap = 3
    numPotSyn = potWidth * potHeight
    numColumns = numColumnRows * numColumnCols

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
                                    numColumnCols,
                                    numColumnRows,
                                    numInputCols,
                                    numInputRows,
                                    centerPotSynapses,
                                    connectedPerm,
                                    minOverlap)

    print "newInputMat = \n%s" % newInputMat
    #potSyn = np.random.rand(1, 1, 4, 4)

    #overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)

    # Return both the overlap values and the inputs from
    # the potential synapses to all columns.
    colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)
    print "len(colOverlaps) = %s" % len(colOverlaps)
    print "colOverlaps = \n%s" % colOverlaps
    print "colPotInputs = \n%s" % colPotInputs

    # limit the overlap values so they are larger then minOverlap
    colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

    print "colOverlaps = \n%s" % colOverlaps


