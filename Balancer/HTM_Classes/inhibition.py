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
        # the minOverlap from the sorted vector of overlaps for each column.
        # This function takes a vector of indicies indicating where the
        # minLocalActivity resides for each row in the matrix.
        # Note: the sorted overlap matrix goes from low to highest so use neg index.
        self.min_OIndex = T.vector(dtype='int32')
        self.s_ColOMat = T.matrix(dtype='float32')
        self.row_numVect = T.vector(dtype='int32')
        self.get_indPosVal = self.s_ColOMat[self.row_numVect, -self.min_OIndex]
        self.get_minLocAct = function([self.min_OIndex,
                                       self.s_ColOMat,
                                       self.row_numVect],
                                      self.get_indPosVal,
                                      allow_input_downcast=True
                                      )

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
                                      self.check_gteq_minLocAct,
                                      on_unused_input='warn',
                                      allow_input_downcast=True
                                      )

        # Create the theano function for calculating
        # a matrix of the columns which should stay active because they
        # won the inhibition convolution for all columns.
        self.col_pat = T.matrix(dtype='int32')
        self.act_cols = T.matrix(dtype='float32')
        self.col_num2 = T.matrix(dtype='int32')
        # A Bug exists if the position at act_cols[0, 0] is not zero.
        # ( This may occur if kernel not centered, centerInhib = 0)
        self.set_winners = self.act_cols[T.switch(T.gt(self.col_pat-1,0),self.col_pat-1,0), T.switch(T.ge(self.col_pat-1,0),self.col_num2,0)]    # self.act_cols[self.col_pat-1, self.col_num]
        #self.get_colwinners = T.switch(T.gt(self.col_pat, 0), self.set_winners, 1)
        self.get_activeColMat = function([self.act_cols,
                                          self.col_pat,
                                          self.col_num2],
                                         self.set_winners,
                                         on_unused_input='warn',
                                         allow_input_downcast=True
                                         )

        # Create the theano function for calculating
        # the sum of the rows from the output of the theano
        # function get_activeColMat. If all the number of elements in a row
        # equal to one then set the col this row represents as active.
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
        # the sum of the rows of the input vector.
        self.in_vect2 = T.vector(dtype='float32')
        self.out_sumvect2 = self.in_vect2.sum(axis=0)
        self.get_sumRowVec = function([self.in_vect2],
                                      self.out_sumvect2,
                                      allow_input_downcast=True)

        # Create the theano function for calculating
        # if the input matrix is larger then 0 (element wise).
        self.in_mat2 = T.matrix(dtype='float32')
        self.lt_zer0 = T.switch(T.gt(self.in_mat2, 0), 1, 0)
        self.get_gtZeroMat = function([self.in_mat2],
                                      self.lt_zer0,
                                      allow_input_downcast=True)

        # Create the theano function for calculating
        # if the input vector is larger then 0 (element wise).
        self.in_vect1 = T.vector(dtype='float32')
        self.gt_zeroVect = T.switch(T.gt(self.in_vect1, 0), 1, 0)
        self.get_gtZeroVect = function([self.in_vect1],
                                       self.gt_zeroVect,
                                       allow_input_downcast=True)

        # Create the theano function for calculating
        # the updated inhibited columns list.
        # A vector is passed in representing which columns have been
        # inhibited, active or not updated yet.
        ## Another input vector represents the winning columns,
        # (this is braodcasted so its size equals colInConvoleList)
        # The last input is the colInConvoleList matrix.
        self.win_columns = T.matrix(dtype='float32')
        self.col_inConvoleMat = T.matrix(dtype='int32')
        self.mat_colNum = T.matrix(dtype='int32')
        self.get_aCols = self.win_columns[self.col_inConvoleMat-1, self.mat_colNum]
        self.check_rCols = T.switch(T.gt(self.col_inConvoleMat, 0), self.get_aCols, 0)
        self.set_inhibCols = function([self.win_columns,
                                       self.col_inConvoleMat,
                                       self.mat_colNum],
                                      self.check_rCols,
                                      allow_input_downcast=True)

        # Create the theano function for calculating
        # the updated inhibiton matrix for the columns.
        # The output is the colInConvoleList where each
        # position represents an inhibited or not col.
        self.inh_colVect = T.vector(dtype='float32')
        self.col_inConvoleMat2 = T.matrix(dtype='int32')
        self.get_upInhibCols = self.inh_colVect[self.col_inConvoleMat2 - 1]
        self.check_gtZero = T.switch(T.gt(self.col_inConvoleMat2, 0), self.get_upInhibCols, 0)
        self.check_vectValue = function([self.inh_colVect,
                                         self.col_inConvoleMat2],
                                        self.check_gtZero,
                                        allow_input_downcast=True)

        # Create the theano function for calculating
        # the first input vector minus the second.
        self.in_vect3 = T.vector(dtype='int32')
        self.in_vect4 = T.vector(dtype='int32')
        self.out_minusvect = self.in_vect3 - self.in_vect4
        self.minus_vect = function([self.in_vect3,
                                    self.in_vect4],
                                   self.out_minusvect,
                                   allow_input_downcast=True)

        # Create the theano function for calculating
        # a matrix of the columns which should stay active because they
        # where not inhibited and won their convole overlap group.
        self.curr_winCols = T.matrix(dtype='int8')
        self.col_pat2 = T.matrix(dtype='int32')
        self.cur_inhib_cols = T.vector(dtype='int32')
        self.col_num3 = T.matrix(dtype='int32')
        self.set_newWinners = T.switch(T.gt(self.cur_inhib_cols[self.col_pat2-1] + self.curr_winCols[self.col_pat2-1, self.col_num3], 1), 1, self.curr_winCols)
        self.get_newColwinners = T.switch(T.ge(self.col_pat2-1, 0), self.set_newWinners, 1)
        self.get_newActiveColMat = function([self.curr_winCols,
                                             self.col_pat2,
                                             self.cur_inhib_cols,
                                             self.col_num3],
                                            self.get_newColwinners,
                                            on_unused_input='warn',
                                            allow_input_downcast=True
                                            )

        # Create the theano function for calculating
        # if a column in the matrix is inhibited.
        # Any inhibited columns should be set as zero.
        # Any columns not iinhibited should be set to the inpu matrix value.
        self.act_cols2 = T.matrix(dtype='float32')
        self.col_pat3 = T.matrix(dtype='int32')
        self.cur_inhib_cols2 = T.vector(dtype='int32')
        self.set_winToZero = T.switch(T.eq(self.cur_inhib_cols2[self.col_pat3-1], 1), 0, self.act_cols2)
        self.check_lZeroCol = T.switch(T.ge(self.col_pat3-1, 0), self.set_winToZero, 0)
        self.check_inhibCols = function([self.act_cols2,
                                         self.col_pat3,
                                         self.cur_inhib_cols2],
                                        self.check_lZeroCol,
                                        allow_input_downcast=True
                                        )

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

        self.colInConvoleList = self.calculateConvolePattern(self.colConvolePatternIndex)
        print "colInConvoleList = \n%s" % self.colInConvoleList

        self.nonPaddingSumVect = self.get_gtZeroMat(self.colInConvoleList)
        self.nonPaddingSumVect = self.get_sumRowMat(self.nonPaddingSumVect)
        print "nonPaddingSumVect = \n%s" % self.nonPaddingSumVect

        # The folowing variables are used for indicies when looking up values
        # in matricies from within a theano function.
        # Create a matrix that just holds the column number for each element
        self.col_num = np.array([[i for i in range(self.potentialWidth*self.potentialHeight)]
                                for j in range(self.width*self.height)])
        # Create just a vector storing the row numbers for each column.
        # This is just an incrementing vector from zero to the number of columns - 1
        self.row_numVect = np.array([i for i in range(self.width*self.height)])

        # Create a vector of minOverlap indicies. This stores the position
        # for each col where the minOverlap resides, in the sorted Convole overlap mat
        self.minOverlapIndex = np.array([self.desiredLocalActivity for i in range(self.width*self.height)])

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

    def calculateActiveCol(self, colOverlapMat):
        # Sort the colOverlap matrix for each row. A row holds the inhib overlap
        # values for a single column.
        sortedColOverlapMat = self.sortOverlapMatrix(colOverlapMat)
        # Get the minLocalActivity for each col.
        minLocalAct = self.get_minLocAct(self.minOverlapIndex,
                                         sortedColOverlapMat,
                                         self.row_numVect)
        #print "minLocalAct = \n%s" % minLocalAct

        # First take the colOverlaps matrix and flatten it into a vector.
        # Broadcast minLocalActivity so it is the same dim as colOverlapMat
        widthColOverlapMat = len(sortedColOverlapMat[0])
        minLocalAct = np.tile(np.array([minLocalAct]).transpose(), (1, widthColOverlapMat))
        # Now calculate for each columns list of overlap values, which ones are larger
        # then the minLocalActivity number.
        activeCols = self.get_activeCol(colOverlapMat, minLocalAct)

        #print "minLocalAct = \n%s" % minLocalAct
        #print "colOverlapMat = \n%s" % colOverlapMat
        #print "activeCols = \n%s" % activeCols

        return activeCols

    def calculateActiveColumnVect(self, activeCols):
        # Calculate for each column a list of columns which that column can
        # be inhibited by. Set the winning columns in this list as one.
        colwinners = self.get_activeColMat(activeCols,
                                           self.colInConvoleList,
                                           self.col_num)

        #print "self.nonPaddingSumVect = \n%s" % self.nonPaddingSumVect
        #print "colwinners = \n%s" % colwinners
        #print "self.colConvolePatternIndex = \n%s" % self.colConvolePatternIndex

        # Now calculate which columns won all their colwinners list.
        # This creates a vector telling us which columns have the highest
        # overlap values and should be active.
        activeColumnVect = self.get_activeColVect(colwinners, self.nonPaddingSumVect)
        #print "activeColumnVect = \n%s" % activeColumnVect

        return activeColumnVect

    def calculateInhibCols(self, activeColumnVect):
        # Now calculate a list of inhibited columns.
        # Create a vector one element for each col. 1 means the col has
        # been inhibited.
        widthColConvolePat = len(self.colConvolePatternIndex[0])
        #print "widthColConvolePat = %s" % widthColConvolePat
        colWinnersMat = np.tile(np.array([activeColumnVect]).transpose(), (1, widthColConvolePat))
        #print "colWinnersMat = \n%s" % colWinnersMat
        #print "colWinnersMat shape w, h = %s,%s" % (len(colWinnersMat[0]), len(colWinnersMat))
        #print "self.colInConvoleList shape w, h = %s,%s" % (len(self.colInConvoleList[0]), len(self.colInConvoleList))

        inhibitedCols = self.set_inhibCols(colWinnersMat,
                                           self.colConvolePatternIndex,
                                           self.col_num)
        #print "inhibitedCols = \n%s" % inhibitedCols
        inhibitedColsVect = self.get_sumRowMat(inhibitedCols)
        #print "inhibitedColsVect = \n%s" % inhibitedColsVect
        updatedInhibCols = self.check_vectValue(inhibitedColsVect, self.colInConvoleList)
        #print "updatedInhibCols = \n%s" % updatedInhibCols
        inhibitedColsVect = self.get_sumRowMat(updatedInhibCols)
        inhibOrActCols = self.get_gtZeroVect(inhibitedColsVect)
        #print "inhibOrActCols = \n%s" % inhibOrActCols
        # Calculate a list of columns that where just inhibited.
        # Just minus the inhibOrActCols vect with the activeColumnVect.
        inhibCols = self.minus_vect(inhibOrActCols, activeColumnVect)
        #print "inhibCols = \n%s" % inhibCols

        # Sum the updatedInhibCols vector and compare to the number of cols
        # If equal then all columns have been inhibited or are active.
        notInhibOrActNum = self.width * self.height - self.get_sumRowVec(inhibOrActCols)
        print "notInhibOrActNum = %s" % notInhibOrActNum

        return inhibCols, notInhibOrActNum

    def calculateWinningCols(self, overlapsGrid):
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

        activeCols = self.calculateActiveCol(colOverlapMat)
        activeColumnVect = self.calculateActiveColumnVect(activeCols)
        inhibCols, notInhibOrActNum = self.calculateInhibCols(activeColumnVect)

        activeColumnVect = activeColumnVect.reshape((self.height, self.width))
        print "activeColumnVect = \n%s" % activeColumnVect
        print "original overlaps = \n%s" % colOverlapGrid

        # If notInhibOrActNum is larger then zero then do the following in a loop:
        # Calculate an updated colWinners matrix by looking at each position in the
        # colOverlapMat and see if that column represented in that position is
        # now inhibited. If so set its overlap value to zero and recalculate the
        # sortedColOverlapMat. Then recalculate the minOverlapIndex for that column.

        #print "self.colConvolePatternIndex = \n%s" % self.colConvolePatternIndex
        #print "old colOverlapMat = \n%s" % colOverlapMat
        colOverlapMat = self.check_inhibCols(colOverlapMat,
                                             self.colConvolePatternIndex,
                                             inhibCols)

        activeCols = self.calculateActiveCol(colOverlapMat)
        activeColumnVect = self.calculateActiveColumnVect(activeCols)
        inhibCols, notInhibOrActNum = self.calculateInhibCols(activeColumnVect)

        # Next recalculate the minOverlap value using the new minOverlapIndex.
        # Rebroadcast this minOverlap index and calculate a new activeCols matrix

        #while notInhibOrActNum > 0:
        # newActiveCols = self.check_inhibCols(activeCols,
        #                                      self.colConvolePatternIndex,
        #                                      inhibCols)

        # newActiveCols needs to recalucalte the minLocalactivity
        # by not including cols that are inhibited.


        # newColWinners = self.get_newActiveColMat(colwinners,
        #                                          self.colInConvoleList,
        #                                          inhibCols,
        #                                          self.col_num)
        # print "newColWinners = \n%s" % newColWinners

        return activeColumnVect

    def sortOverlapMatrix(self, colOverlapVals):
        # colOverlapVals, each row is a list of overlaps values that
        # a column can potentially inhibit.
        # Sort the grid of overlap values from largest to
        # smallest for each columns inhibit overlap vect.
        sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
        #print "sortedColOverlapsVals = \n%s" % sortedColOverlapsVals
        return sortedColOverlapsVals


if __name__ == '__main__':

    potWidth = 2
    potHeight = 2
    centerInhib = 1
    numRows = 4
    numCols = 5
    desiredLocalActivity = 1

     # Some made up inputs to test with
    colOverlapGrid = np.random.randint(10, size=(numRows, numCols))

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns
    print "original overlaps = \n%s" % colOverlapGrid

    #print "activeColumns = \n%s" % activeColumns


