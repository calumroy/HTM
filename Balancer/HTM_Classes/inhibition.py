import theano.tensor as T
from theano import function, shared
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import tensor
from theano.tensor import set_subtensor

from theano.tensor.sort import argsort, sort
from theano import Mode
import math


'''
A class to calculate the inhibition of columns for a HTM layer.
This class uses theano functions to speed up the computation.
It can be implemented on a GPU see theano documents for
enabling GPU calculations.

Inputs:
It uses the overlap values for each column, expressed in matrix form.
It must be in a matrix so convolution can be used to determine
column neighbours.


Outputs:
It outputs a binary vector where each position indicates if a column
is active or not.
'''


class inhibitionCalculator():
    def __init__(self, width, height, potentialInhibWidth, potentialInhibHeight,
                 desiredLocalActivity, centerInhib=1):
        # Temporal Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerInhib = centerInhib
        self.width = width
        self.height = height
        self.potentialWidth = potentialInhibWidth
        self.potentialHeight = potentialInhibHeight
        self.areaKernel = self.potentialWidth * self.potentialHeight
        self.desiredLocalActivity = desiredLocalActivity
        # Store how much padding is added to the input grid
        self.topPos_y = 0
        self.bottomPos_y = 0
        self.leftPos_x = 0
        self.rightPos_x = 0

        # Create theano variables and functions
        ############################################
        # Create the theano function for calculating
        # the addition of a small tie breaker value to each overlap value.
        self.o_grid = T.matrix(dtype='float32')
        self.tie_grid = T.matrix(dtype='float32')
        self.add_vals = T.add(self.o_grid, self.tie_grid)
        self.add_tieBreaker = function([self.o_grid, self.tie_grid],
                                       self.add_vals,
                                       on_unused_input='warn',
                                       allow_input_downcast=True)

        # Create the theano function for calculating
        # the inputs to a column from an input grid.
        self.kernalSize = (self.potentialHeight, self.potentialWidth)
        # poolstep is how far to move the kernal in each direction.
        self.poolstep = (1, 1)
        # Create the theano function for calculating the overlaps of
        # the potential columns that any column can inhibit.
        self.neib_shape = T.as_tensor_variable(self.kernalSize)
        self.neib_step = T.as_tensor_variable(self.poolstep)
        self.pool_inp = T.tensor4('pool_input', dtype='float32')
        self.pool_convole = images2neibs(self.pool_inp, self.neib_shape, self.neib_step, mode='valid')
        self.pool_inputs = function([self.pool_inp],
                                    self.pool_convole,
                                    on_unused_input='warn',
                                    allow_input_downcast=True)

        # Create the theano function for calculating
        # the sorted vector of overlaps for each columns inhib overlaps
        self.o_mat = tensor.dmatrix()
        #self.so_mat = tensor.dmatrix()
        self.axis = tensor.scalar()
        self.arg_sort = sort(self.o_mat, self.axis, "quicksort")
        self.sort_vect = function([self.o_mat, self.axis], self.arg_sort)

        # Create the theano function for calculating
        # if a column should be active or not based on whether it
        # has an overlap greater then or equal to the minLocalActivity.
        self.minLocalActivity = T.matrix(dtype='float32')
        self.colOMat = T.matrix(dtype='float32')
        self.check_gt_zero = T.switch(T.gt(self.colOMat, 0), 1, 0)
        self.check_gteq_minLocAct = T.switch(T.ge(self.colOMat, self.minLocalActivity), self.check_gt_zero, 0)
        #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
        self.get_activeCol = function([self.colOMat,
                                      self.minLocalActivity],
                                      [self.check_gteq_minLocAct],
                                      on_unused_input='warn',
                                      allow_input_downcast=True
                                      )

        # Create the theano function for calculating
        # a vector of the columns which should stay active because they
        # won the inhibition convolution for all columns.
        self.col_pat = T.matrix(dtype='int8')
        self.act_cols = T.matrix(dtype='float32')
        self.col_num = T.matrix(dtype='int8')
        self.win_colconvole = T.matrix(dtype='float32')
        self.set_winners = self.act_cols[self.col_pat-1, self.col_num]    # self.act_cols[self.col_pat-1, self.col_num]
        #self.get_colwinners = T.switch(T.gt(self.col_pat, 0), self.set_winners, 1)
        self.get_activeColMat = function([self.act_cols,
                                          self.col_pat,
                                          self.col_num],
                                         self.set_winners,
                                         on_unused_input='warn',
                                         allow_input_downcast=True
                                         )

        # Create the theano function for calculating
        # the sum of the rows from the output of the theano
        # function get_activeColVect. If all the rows are zero
        # then the col this row represents should be set as active.
        self.col_winConPat = T.matrix(dtype='float32')
        self.non_padSum = T.vector(dtype='float32')
        self.w_cols = self.col_winConPat.sum(axis=1)
        self.test_lcol = T.switch(T.eq(self.w_cols, self.non_padSum), 1, 0)
        self.get_activeColVect = function([self.col_winConPat,
                                           self.non_padSum],
                                          self.test_lcol,
                                          allow_input_downcast=True)

        # Create the theano function for calculating
        # the sum of the rows of the input matrix.
        self.in_mat1 = T.matrix(dtype='float32')
        self.out_summat2 = self.in_mat1.sum(axis=1)
        self.get_sumRowMat = function([self.in_mat1],
                                      self.out_summat2,
                                      allow_input_downcast=True)

        # Create the theano function for calculating
        # if the input matrix is larger then 0 (element wise).
        self.in_mat2 = T.matrix(dtype='float32')
        self.lt_zer0 = T.switch(T.gt(self.in_mat2, 0), 1, 0)
        self.get_gtZeroMat = function([self.in_mat2],
                                      self.lt_zer0,
                                      allow_input_downcast=True)

        #### END of Theano functions and variables definitions
        #################################################################
        # Now Also calcualte a convole grid so the columns position
        # in the resulting col inhib overlap matrix can be tracked.
        self.incrementingMat = np.array([[1+i+self.width*j for i in range(self.width)] for j in range(self.height)])
        self.colConvolePatternIndex = self.getColInhibInputs(self.incrementingMat)
        print "colConvole = \n%s" % self.colConvolePatternIndex

        # Calculate a matrix storing the location of the numbers from
        # colConvolePatternIndex.
        unConvoleTestIn = np.array(
            [[0,0,0,1],
             [0,0,1,2],
             [0,0,2,3],
             [0,0,3,4],
             [0,0,4,5],
             [0,1,0,6],
             [1,2,6,7]])
        print "test unconvole = \n%s" % self.calculateConvolePattern(unConvoleTestIn)

        self.unConvolePattern = self.calculateConvolePattern(self.colConvolePatternIndex)
        print "unConvolePattern = \n%s" % self.unConvolePattern

        self.nonPaddingSumVect = self.get_gtZeroMat(self.unConvolePattern)
        self.nonPaddingSumVect = self.get_sumRowMat(self.nonPaddingSumVect)
        print "nonPaddingSumVect = \n%s" % self.nonPaddingSumVect

    def calculateConvolePattern(self, inputGrid):
        '''
        Determine the row number locations of the column
        numbers in the inputGrid.

        eg.
        inputGrid                   Calculated output
        [[  0.   0.   0.   1.]      [[  7.   6.   2.   1.]
         [  0.   0.   1.   2.]       [  0.   7.   3.   2.]
         [  0.   0.   2.   3.]       [  0.   0.   4.   3.]
         [  0.   0.   3.   4.]       [  0.   0.   5.   4.]
         [  0.   0.   4.   5.]       [  0.   0.   0.   5.]
         [  0.   1.   0.   6.]       [  0.   0.   7.   6.]
         [  1.   2.   6.   7.]]      [  0.   0.   0.   7.]]

         Note: height = numCols = self.width * self.height
        '''

        width = len(inputGrid[0])
        height = len(inputGrid)

        outputGrid = np.array([[0 for i in range(width)] for j in range(height)])

        #print "width = %s height = %s" % (width, height)
        curColNum = 0
        curRowNum = 0
        for c in range(width):
            # Search for the column numbers.
            # They are always in order down the column
            curColNum = 1
            curRowNum = 0
            for r in range(height):
                if inputGrid[r, c] > curColNum:
                    curRowNum = curRowNum + inputGrid[r, c] - curColNum
                    curColNum = inputGrid[r, c]

                if inputGrid[r, c] == curColNum:
                    curColNum += 1
                    curRowNum += 1
                    outputGrid[inputGrid[r, c]-1, c] = r+1

        return outputGrid

    def addPaddingToInput(self, inputGrid):

        if self.centerInhib == 0:
            # The potential inhibited columns are not centered around each column.
            # This means only the right side and bottom of the input
            # need padding.
            self.topPos_y = 0
            self.bottomPos_y = self.potentialHeight-1
            self.leftPos_x = 0
            self.rightPos_x = self.potentialWidth-1
        else:
            # The otential inhibited columns are centered over the column
            # This means all sides of the input need padding.
            self.topPos_y = self.potentialHeight/2
            self.bottomPos_y = int(math.ceil(self.potentialHeight/2.0))-1
            self.leftPos_x = self.potentialWidth/2
            self.rightPos_x = int(math.ceil(self.potentialWidth/2.0))-1

        # Make sure all are larger then zero still
        if self.topPos_y < 0:
            self.topPos_y = 0
        if self.bottomPos_y < 0:
            self.bottomPos_y = 0
        if self.leftPos_x < 0:
            self.leftPos_x = 0
        if self.rightPos_x < 0:
            self.rightPos_x = 0

        # Add the padding around the edges of the input.
        inputGrid = np.lib.pad(inputGrid,
                              ((0, 0),
                               (0, 0),
                               (self.topPos_y, self.bottomPos_y),
                               (self.leftPos_x, self.rightPos_x)),
                               'constant',
                               constant_values=(0))

        #print "inputGrid = \n%s" % inputGrid

        return inputGrid

    def getColInhibInputs(self, inputGrid):
        # Take the input and put it into a 4D tensor.
        # This is because the theano function images2neibs
        # works with 4D tensors only.
        inputGrid = np.array([[inputGrid]])

        print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # work out how much padding is needed on the borders
        # using the defined potential inhib width and potential inhib height.
        inputGrid = self.addPaddingToInput(inputGrid)

        # Calculate the input overlaps for each column.
        inputInhibCols = self.pool_inputs(inputGrid)
        # The returned array is within a list so just use pos 0.
        print "inputInhibCols = \n%s" % inputInhibCols
        print "inputInhibCols.shape = %s,%s" % inputInhibCols.shape
        return inputInhibCols

    def calculateInhibCols(self, overlapsGrid):
        # Take the overlapsGrid and calulate a binary list
        # describing the active columns ( 1 is active, 0 not active).

        height, width = overlapsGrid.shape
        numCols = width * height
        print " width, height, numCols = %s, %s, %s" % (width, height, numCols)
        print "overlapsGrid = \n%s" % overlapsGrid
        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        normValue = 1.0/float(numCols)
        tieBreaker = np.array([[(i+j*width)*normValue for i in range(width)] for j in range(height)])
        print "tieBreaker = \n%s" % tieBreaker
        # Add the tieBreaker value to the overlap values.
        overlapsGrid = self.add_tieBreaker(overlapsGrid, tieBreaker)
        print "overlapsGrid = \n%s" % overlapsGrid
        # Calculate the overlaps associated with columns that can be inhibited.
        colOverlapMat = self.getColInhibInputs(overlapsGrid)
        print "colOverlapMat = \n%s" % colOverlapMat
        # Sort the colOverlap matrix for each row. A row hold the inhib overlap
        # values for a single column.
        sortedColOverlapMat = self.sortOverlapMatrix(colOverlapMat)
        # Get the minLocalActivity for each col.
        # Plus one because of the range.
        minOverlapIndex = self.desiredLocalActivity
        # check to make sure minOverlapIndex is smaller then the width of
        # the sortedColOverlapMat matrix.
        if minOverlapIndex > len(sortedColOverlapMat[0]):
            minOverlapIndex = len(sortedColOverlapMat[0])
        minLocalAct = sortedColOverlapMat[:, -(minOverlapIndex)]
        print "minLocalAct = \n%s" % minLocalAct

        # First take the colOverlaps matrix and flatten it into a vector.
        # Broadcast minLocalActivity so it is the same dim as colOverlapMat
        widthColOverlapMat = len(sortedColOverlapMat[0])
        minLocalAct = np.tile(np.array([minLocalAct]).transpose(), (1, widthColOverlapMat))
        # Now calculate for each columns list of overlap values, which are larger
        # then the minLocalActivity number.
        activeCols = self.get_activeCol(colOverlapMat, minLocalAct)
        activeCols = activeCols[0]

        print "minLocalAct = \n%s" % minLocalAct
        print "colOverlapMat = \n%s" % colOverlapMat
        print "activeCols = \n%s" % activeCols
        print "self.incrementingMat = \n%s" % self.incrementingMat
        #print "self.colConvolePatternIndex = \n%s" % self.colConvolePatternIndex
        print "self.unConvolePattern = \n%s" % self.unConvolePattern

        col_num = np.array([[i for i in range(self.potentialWidth*self.potentialHeight)]
                            for j in range(self.width*self.height)])
        print "col_num = \n%s" % col_num

        colwinners = self.get_activeColMat(activeCols,
                                           self.unConvolePattern,
                                           col_num)

        print "self.nonPaddingSumVect = \n%s" % self.nonPaddingSumVect
        print "colwinners = \n%s" % colwinners

        activeColumnVect = self.get_activeColVect(colwinners, self.nonPaddingSumVect)
        print "activeColumnVect = \n%s" % activeColumnVect

        return activeColumnVect

    def sortOverlapMatrix(self, colOverlapVals):
        # colOverlapVals, each row is a list of overlaps values that
        # a column can potentially inhibit.
        # Sort the grid of overlap values from largest to
        # smallest for each columns inhibit overlap vect.
        sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
        print "sortedColOverlapsVals = \n%s" % sortedColOverlapsVals
        return sortedColOverlapsVals


if __name__ == '__main__':

    potWidth = 2
    potHeight = 2
    centerInhib = 1
    numRows = 5
    numCols = 4
    desiredLocalActivity = 1

     # Some made up inputs to test with
    colOverlapGrid = np.random.randint(10, size=(numRows, numCols))

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    activeColumns = inhibCalculator.calculateInhibCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns

    print "original overlaps = \n%s" % colOverlapGrid
    #print "activeColumns = \n%s" % activeColumns


