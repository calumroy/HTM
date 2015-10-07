import theano.tensor as T
from theano import function, shared
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import tensor
from theano.tensor import set_subtensor

from theano.tensor.sort import argsort, sort
from theano import Mode
import math

import cProfile

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

THIS IS A REINIMPLEMENTATION OF THE OLD INHIBITON CODE BELOW

    #print "length active columns before deleting = %s" % len(self.activeColumns)
    self.activeColumns = np.array([], dtype=object)
    #print "actve cols before %s" %self.activeColumns
    allColumns = self.columns.flatten().tolist()
    # Get all the columns in a 1D array then sort them based on their overlap value.
    #allColumns = allColumns[np.lexsort(allColumns.overlap, axis=None)]
    allColumns.sort(key=lambda x: x.overlap, reverse=True)
    # Now start from the columns with the highest overlap and inhibit
    # columns with smaller overlaps.
    for c in allColumns:
        if c.overlap > 0:
            # Get the neighbours of the column
            neighbourCols = self.neighbours(c)
            minLocalActivity = self.kthScore(neighbourCols, self.desiredLocalActivity)
            #print "current column = (%s, %s) overlap = %d min = %d" % (c.pos_x, c.pos_y,
            #                                                            c.overlap, minLocalActivity)
            if c.overlap > minLocalActivity:
                self.activeColumns = np.append(self.activeColumns, c)
                self.columnActiveAdd(c, timeStep)
                # print "ACTIVE COLUMN x,y = %s, %s overlap = %d min = %d" % (c.pos_x, c.pos_y,
                #                                                             c.overlap, minLocalActivity)
            elif c.overlap == minLocalActivity:
                # Check the neighbours and see how many have an overlap
                # larger then the minLocalctivity or are already active.
                # These columns will be set active.
                numActiveNeighbours = 0
                for d in neighbourCols:
                    if (d.overlap > minLocalActivity or self.columnActiveState(d, self.timeStep) is True):
                        numActiveNeighbours += 1
                # if less then the desired local activity have been set
                # or will be set as active then activate this column as well.
                if numActiveNeighbours < self.desiredLocalActivity:
                    #print "Activated column x,y = %s, %s numActiveNeighbours = %s" % (c.pos_x, c.pos_y, numActiveNeighbours)
                    self.activeColumns = np.append(self.activeColumns, c)
                    self.columnActiveAdd(c, timeStep)
                else:
                    # Set the overlap score for the losing columns to zero
                    c.overlap = 0
            else:
                # Set the overlap score for the losing columns to zero
                c.overlap = 0
        self.updateActiveDutyCycle(c)
        # Update the active duty cycle variable of every column
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
        # if the colInConvole matrix. This takes a vector
        # storing an offset number and adds this to the input
        # matrix if the element in the input matrix is greater then
        # zero.
        self.in_colPatMat = T.matrix(dtype='int32')
        self.in_colAddVect = T.vector(dtype='int32')
        self.in_colNegVect = T.vector(dtype='int32')
        self.col_num3 = T.matrix(dtype='int32')
        # self.row_numMat5 = T.matrix(dtype='int32')
        self.check_gtZero2 = T.switch(T.gt(self.in_colPatMat, 0),
                                     (self.in_colPatMat +
                                      self.in_colAddVect[self.col_num3] -
                                      self.in_colNegVect[self.col_num3]+1),
                                      0)
        self.add_toConvolePat = function([self.in_colPatMat,
                                          self.in_colAddVect,
                                          self.in_colNegVect,
                                          self.col_num3],
                                         self.check_gtZero2,
                                         allow_input_downcast=True)

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
        self.row_numVect2 = T.vector(dtype='int32')
        self.get_indPosVal = self.s_ColOMat[self.row_numVect2, -self.min_OIndex]
        self.get_minLocAct = function([self.min_OIndex,
                                       self.s_ColOMat,
                                       self.row_numVect2],
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
        # if a column is inhibited then set that location to one only if
        # that row does not represent that inhibited column.
        self.col_pat = T.matrix(dtype='int32')
        self.act_cols = T.matrix(dtype='float32')
        self.col_num2 = T.matrix(dtype='int32')
        self.row_numMat4 = T.matrix(dtype='int32')
        self.cur_inhib_cols4 = T.vector(dtype='int32')

        self.test_meInhib = T.switch(T.eq(self.cur_inhib_cols4[self.row_numMat4], 1), 0, 1)
        self.set_winners = self.act_cols[self.col_pat-1, self.col_num2]
        self.check_colNotInhib = T.switch(T.lt(self.cur_inhib_cols4[self.col_pat-1], 1), self.set_winners, self.test_meInhib)
        self.check_colNotPad = T.switch(T.ge(self.col_pat-1, 0), self.check_colNotInhib, 0)
        self.get_activeColMat = function([self.act_cols,
                                          self.col_pat,
                                          self.col_num2,
                                          self.row_numMat4,
                                          self.cur_inhib_cols4],
                                         self.check_colNotPad,
                                         on_unused_input='warn',
                                         allow_input_downcast=True
                                         )

        # Create the theano function for calculating
        # the rows that have more then or equal to
        # the input non_padSum. If this is true then set
        # in the output vector the col this row represents as active.
        # This function calculates if a column beat all the other non inhibited
        # columns in the convole overlap groups.
        self.col_winConPat = T.matrix(dtype='float32')
        self.non_padSum = T.vector(dtype='float32')
        self.w_cols = self.col_winConPat.sum(axis=1)
        self.test_lcol = T.switch(T.ge(self.w_cols, self.non_padSum), 1, 0)
        self.test_gtZero = T.switch(T.gt(self.non_padSum, 0), self.test_lcol, 0)
        self.get_activeColVect = function([self.col_winConPat,
                                           self.non_padSum],
                                          self.test_gtZero,
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
        # the list of columns that are not active but
        # contain an active column in their convole inhib list.
        # A vector is passed in representing which columns have been
        # inhibited, active or not updated yet.
        self.act_cols4 = T.vector(dtype='float32')
        #self.win_columns = T.matrix(dtype='float32')
        self.col_convolePatInd = T.matrix(dtype='int32')
        self.row_numMat3 = T.matrix(dtype='int32')
        #self.get_aCols = self.win_columns[self.col_convolePatInd-1, self.mat_colNum]
        self.get_aCols = T.switch(T.gt(self.act_cols4[self.row_numMat3], 0),
                                  0, self.act_cols4[self.col_convolePatInd - 1])
        self.check_rCols = T.switch(T.gt(self.col_convolePatInd, 0), self.get_aCols, 0)
        self.set_inhibCols = function([self.col_convolePatInd,
                                       self.row_numMat3,
                                       self.act_cols4],
                                      self.check_rCols,
                                      allow_input_downcast=True)

        # Create the theano function for calculating
        # the updated inhibiton matrix for the columns.
        # The output is the colInConvoleList where each
        # position represents an inhibited or not col.
        #self.inh_colVect = T.vector(dtype='float32')
        self.act_cols3 = T.vector(dtype='float32')
        self.col_inConvoleMat2 = T.matrix(dtype='int32')
        self.row_numMat2 = T.matrix(dtype='int32')
        self.get_upInhibCols = T.switch(T.gt(self.act_cols3[self.row_numMat2], 0),
                                        0, self.act_cols3[self.col_inConvoleMat2 - 1])
        self.check_gtZero = T.switch(T.gt(self.col_inConvoleMat2, 0), self.get_upInhibCols, 0)
        self.check_vectValue = function([self.col_inConvoleMat2,
                                         self.act_cols3,
                                         self.row_numMat2],
                                        self.check_gtZero,
                                        allow_input_downcast=True)

        # Create the theano function for calculating
        # if a column should be inhibited because the column
        # has a zero overlap value.
        self.col_overlapVect = T.vector(dtype='float32')
        self.col_inhib = T.vector(dtype='int32')
        self.check_ltOne = T.switch(T.lt(self.col_overlapVect, 1), 1, self.col_inhib)
        self.inhibit_zeroOverlap = function([self.col_overlapVect,
                                             self.col_inhib],
                                            self.check_ltOne,
                                            allow_input_downcast=True)
        # Create the theano function for calculating
        # if a column should not be active because the column
        # has a zero overlap value.
        self.col_overlapVect = T.vector(dtype='float32')
        self.col_active = T.vector(dtype='int32')
        self.check_ltOne = T.switch(T.lt(self.col_overlapVect, 1), 0, self.col_active)
        self.disable_zeroOverlap = function([self.col_overlapVect,
                                             self.col_active],
                                            self.check_ltOne,
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
        # the first input vector pls the second.
        self.in_vect5 = T.vector(dtype='int32')
        self.in_vect6 = T.vector(dtype='int32')
        self.out_sumvect = self.in_vect5 + self.in_vect6
        self.sum_vect = function([self.in_vect5,
                                  self.in_vect6],
                                 self.out_sumvect,
                                 allow_input_downcast=True)

        # Create the theano function for calculating
        # if a column in the matrix is inhibited.
        # Any inhibited columns should be set as zero.
        # Any columns not inhibited should be set to the input matrix value.
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

        # Create the theano function for calculating
        # if a column in the matrix is inhibited.
        # Any inhibited columns should be set as zero.
        # Any columns not inhibited should be set to the input pattern matrix value.
        self.col_pat4 = T.matrix(dtype='int32')
        self.cur_inhib_cols3 = T.vector(dtype='int32')
        self.set_patToZero = T.switch(T.eq(self.cur_inhib_cols3[self.col_pat4-1], 1), 0, self.col_pat4)
        self.check_lZeroCol2 = T.switch(T.ge(self.col_pat4-1, 0), self.set_patToZero, 0)
        self.check_inhibColsPat = function([self.col_pat4,
                                            self.cur_inhib_cols3],
                                           self.check_lZeroCol2,
                                           allow_input_downcast=True
                                           )

        #### END of Theano functions and variables definitions
        #################################################################
        # The folowing variables are used for indicies when looking up values
        # in matricies from within a theano function.
        # Create a matrix that just holds the column number for each element
        self.col_num = np.array([[i for i in range(self.potentialWidth*self.potentialHeight)]
                                for j in range(self.width*self.height)])

        # Create a matrix that just holds the row number for each element
        self.row_numMat = np.array([[j for i in range(self.potentialWidth*self.potentialHeight)]
                                   for j in range(self.width*self.height)])

        # Create just a vector storing the row numbers for each column.
        # This is just an incrementing vector from zero to the number of columns - 1
        self.row_numVect = np.array([i for i in range(self.width*self.height)])

        # Create just a vector stroing if a column is inhibited or not
        self.inhibCols = np.array([0 for i in range(self.width*self.height)])

        # Create a vector of minOverlap indicies. This stores the position
        # for each col where the minOverlap resides, in the sorted Convole overlap mat
        self.minOverlapIndex = np.array([self.desiredLocalActivity for i in range(self.width*self.height)])

        # Now Also calcualte a convole grid so the columns position
        # in the resulting col inhib overlap matrix can be tracked.
        self.incrementingMat = np.array([[1+i+self.width*j for i in range(self.width)] for j in range(self.height)])
        #print "self.incrementingMat = \n%s" % self.incrementingMat
        #print "potential height, width = %s, %s " %(self.potentialHeight, self.potentialWidth)
        self.colConvolePatternIndex = self.getColInhibInputs(self.incrementingMat)
        #print "colConvole = \n%s" % self.colConvolePatternIndex
        #print "colConvole height, width = %s, %s " % (len(self.colConvolePatternIndex),len(self.colConvolePatternIndex[0]))

        # Calculate a matrix storing the location of the numbers from
        # colConvolePatternIndex.
        self.colInConvoleList = self.calculateConvolePattern(self.colConvolePatternIndex)
        #print "colInConvoleList = \n%s" % self.colInConvoleList

        self.nonPaddingSumVect = self.get_gtZeroMat(self.colInConvoleList)
        self.nonPaddingSumVect = self.get_sumRowMat(self.nonPaddingSumVect)
        #print "nonPaddingSumVect = \n%s" % self.nonPaddingSumVect

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

        #print "inputGrid = \n%s" % inputGrid
        width = len(inputGrid[0])
        height = len(inputGrid)

        rolledInputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        outputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        firstNonZeroIndVect = np.array([0 for i in range(width)])
        firstNonZeroVect = np.array([0 for i in range(width)])

        #print "width = %s height = %s" % (width, height)
        print "Setting Up theano inhibition calculator"
        for c in range(width):
            #print "c = %s" % c
            # Search for the column numbers.
            # They are always in order down the column
            # Now roll each column in the inputGrid upwards by the
            # this is a negative numpy roll.
            for r in range(height):
                firstNonZero = int(inputGrid[r, c])
                if firstNonZero > 0.0:
                    firstNonZeroIndVect[c] = r
                    firstNonZeroVect[c] = firstNonZero
                    rolledInputGrid[:, c] = np.roll(inputGrid[:, c], (-r+firstNonZero-1), axis=0)
                    break
        print "Done"

        #print "inputGrid = \n%s" % inputGrid

        #print "firstNonZeroIndVect = \n%s" % firstNonZeroIndVect
        #print "firstNonZeroVect = \n%s" % firstNonZeroVect

        outputGrid = self.add_toConvolePat(rolledInputGrid,
                                           firstNonZeroIndVect,
                                           firstNonZeroVect,
                                           self.col_num)

        #print "outputGrid = \n%s" % outputGrid

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

        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # work out how much padding is needed on the borders
        # using the defined potential inhib width and potential inhib height.
        inputGrid = self.addPaddingToInput(inputGrid)

        #print "padded inputGrid = \n%s" % inputGrid
        # Calculate the input overlaps for each column.
        inputInhibCols = self.pool_inputs(inputGrid)
        # The returned array is within a list so just use pos 0.
        #print "inputInhibCols = \n%s" % inputInhibCols
        #print "inputInhibCols.shape = %s,%s" % inputInhibCols.shape
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

        #print "colOverlapMat = \n%s" % colOverlapMat
        # print "sortedColOverlapMat = \n%s" % sortedColOverlapMat
        # print "minLocalAct = \n%s" % minLocalAct
        # print "activeCols = \n%s" % activeCols

        return activeCols

    def calculateActiveColumnVect(self, activeCols, inhibCols, colOverlapVect):
        # Calculate for each column a list of columns which that column can
        # be inhibited by. Set the winning columns in this list as one.
        # If a column is inhibited already then all those positions in
        # the colwinners relating to that col are set as one. This means
        # the inhibited columns don't determine the active columns
        colwinners = self.get_activeColMat(activeCols,
                                           self.colInConvoleList,
                                           self.col_num,
                                           self.row_numMat,
                                           inhibCols)

        # # Calculate for each row the number it should sum to if the col won all
        # # of it's convole inhib groups (excluding the columns that have been inhibited).
        # nonPadOrInhibSumVect = self.check_inhibColsPat(self.colInConvoleList,
        #                                                inhibCols)

        # nonPadOrInhibSumVect = self.get_gtZeroMat(nonPadOrInhibSumVect)
        # nonPadOrInhibSumVect = self.get_sumRowMat(nonPadOrInhibSumVect)
        # print "nonPadOrInhibSumVect = \n%s" % nonPadOrInhibSumVect

        # print "self.nonPaddingSumVect = \n%s" % self.nonPaddingSumVect
        # print "self.colInConvoleList = \n%s" % self.colInConvoleList
        # print "inhibCols = \n%s" % inhibCols
        # print "colwinners = \n%s" % colwinners

        # Now calculate which columns won all their colwinners list.
        # This creates a vector telling us which columns have the highest
        # overlap values and should be active. Make sure the self.nonPaddingSumVect is not zero.
        activeColumnVect = self.get_activeColVect(colwinners, self.nonPaddingSumVect)

        # If the column has a zero overlap value (ie its overlap value
        # plus the tiebreaker is less then one then do not allow it to be active.
        activeColumnVect = self.disable_zeroOverlap(colOverlapVect,
                                                    activeColumnVect)
        #print "activeColumnVect = \n%s" % activeColumnVect

        return activeColumnVect

    def calculateInhibCols(self, activeColumnVect, colOverlapVect):
        # Now calculate a list of inhibited columns.
        # Create a vector one element for each col. 1 means the col has
        # been inhibited.
        widthColConvolePat = len(self.colConvolePatternIndex[0])
        #print "widthColConvolePat = %s" % widthColConvolePat
        colWinnersMat = np.tile(np.array([activeColumnVect]).transpose(), (1, widthColConvolePat))
        #print "colWinnersMat = \n%s" % colWinnersMat
        #print "colWinnersMat shape w, h = %s,%s" % (len(colWinnersMat[0]), len(colWinnersMat))
        #print "self.colInConvoleList shape w, h = %s,%s" % (len(self.colInConvoleList[0]), len(self.colInConvoleList))

        # Calculates which columns convole inhib group contains active columns.
        # Do not include columns that are active.
        inhibitedCols = self.set_inhibCols(self.colConvolePatternIndex,
                                           self.row_numMat,
                                           activeColumnVect)
        #print "inhibitedCols = \n%s" % inhibitedCols
        inhibitedColsVect = self.get_sumRowMat(inhibitedCols)
        #print "inhibitedColsVect = \n%s" % inhibitedColsVect
        # Now also calculate which columns were in the convole groups
        # of the active cols and should therfore be inhibited.
        # If the column is active do not include it.
        updatedInhibCols = self.check_vectValue(self.colInConvoleList,
                                                activeColumnVect,
                                                self.row_numMat)

        # Now if a column was in the convole group of an active column then it should be inhibited.
        inhibitedColsVect2 = self.get_sumRowMat(updatedInhibCols)

        # The list of the inhibited cols is the active inhibited cols in
        # inhibitedColsVect and inhibitedColsVect2.
        inhibColsVector = self.sum_vect(inhibitedColsVect, inhibitedColsVect2)
        # Now see which columns appeared in either list of inhibited columns
        inhibCols = self.get_gtZeroVect(inhibColsVector)

        # If the column has a zero overlap value (ie its overlap value
        # plus the tiebreaker is less then one then inhibit the column.
        inhibCols = self.inhibit_zeroOverlap(colOverlapVect,
                                             inhibCols)

        #print "inhibCols = \n%s" % inhibCols

        #print "inhibOrActCols = \n%s" % inhibOrActCols
        # Calculate a list of columns that where just inhibited.
        # Just minus the inhibOrActCols vect with the activeColumnVect.
        #inhibCols = self.minus_vect(inhibOrActCols, activeColumnVect)
        #print "inhibCols = \n%s" % inhibCols
        #print "reshaped inhibCols = \n%s" % inhibCols.reshape((self.height, self.width))

        # Sum the updatedInhibCols vector and compare to the number of cols
        # If equal then all columns have been inhibited or are active.
        notInhibOrActNum = self.width * self.height - self.get_sumRowVec(inhibCols) - self.get_sumRowVec(activeColumnVect)
        #print "notInhibOrActNum = %s" % notInhibOrActNum

        return inhibCols, notInhibOrActNum

    def calculateWinningCols(self, overlapsGrid):
        # Take the overlapsGrid and calulate a binary list
        # describing the active columns ( 1 is active, 0 not active).

        height, width = overlapsGrid.shape
        numCols = width * height

        # Reset the inhibited columns vector. All columns start
        # as uninhibited. This sets every element as zero.
        self.inhibCols[:] = 0

        #print " width, height, numCols = %s, %s, %s" % (width, height, numCols)
        #print "overlapsGrid = \n%s" % overlapsGrid

        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        normValue = 1.0/float(numCols+1)
        tieBreaker = np.array([[(1+i+j*width)*normValue for i in range(width)] for j in range(height)])
        #print "tieBreaker = \n%s" % tieBreaker
        # Add the tieBreaker value to the overlap values.
        overlapsGridTie = self.add_tieBreaker(overlapsGrid, tieBreaker)
        #print "overlapsGridTie = \n%s" % overlapsGridTie
        # Calculate the overlaps associated with columns that can be inhibited.
        colOverlapMatOrig = self.getColInhibInputs(overlapsGridTie)

        # Create a vector of the overlap values for each column
        colOverlapVect = overlapsGridTie.flatten()
        #print "colOverlapVect = \n%s" % colOverlapVect

        activeCols = self.calculateActiveCol(colOverlapMatOrig)
        #print "before updating self.inhibCols \n%s" % self.inhibCols
        activeColumnVect = self.calculateActiveColumnVect(activeCols, self.inhibCols, colOverlapVect)
        #print "activeColumnVect = \n%s" % activeColumnVect
        self.inhibCols, notInhibOrActNum = self.calculateInhibCols(activeColumnVect, colOverlapVect)
        #print "self.inhibCols \n%s" % self.inhibCols

        activeColumns = activeColumnVect.reshape((self.height, self.width))
        #print "activeColumns = \n%s" % activeColumns
        #print "original overlaps = \n%s" % overlapsGrid

        # activeColumnVect = activeColumnVect.reshape((self.height, self.width))
        # print "activeColumnVect = \n%s" % activeColumnVect
        # print "original overlaps = \n%s" % colOverlapGrid

        # If notInhibOrActNum is larger then zero then do the following in a loop:
        # Calculate an updated colWinners matrix by looking at each position in the
        # colOverlapMatOrig and see if that column represented in that position is
        # now inhibited. If so set its overlap value to zero and recalculate the
        # sortedColOverlapMat. Then recalculate the minOverlapIndex for that column.

        #print "self.colConvolePatternIndex = \n%s" % self.colConvolePatternIndex
        #print "old colOverlapMatOrig = \n%s" % colOverlapMatOrig
        loopedTimes = 0
        while notInhibOrActNum > 0:
            loopedTimes += 1
            #print "colOverlapMatOrig = \n%s" % colOverlapMatOrig
            #print "self.colConvolePatternIndex = \n%s" % self.colConvolePatternIndex
            #print "self.inhibCols \n%s" % self.inhibCols
            colOverlapMat = self.check_inhibCols(colOverlapMatOrig,
                                                 self.colConvolePatternIndex,
                                                 self.inhibCols)
            #print "colOverlapMat = \n%s" % colOverlapMat

            activeCols = self.calculateActiveCol(colOverlapMat)
            activeColumnVect = self.calculateActiveColumnVect(activeCols, self.inhibCols, colOverlapVect)
            self.inhibCols, notInhibOrActNum = self.calculateInhibCols(activeColumnVect, colOverlapVect)

            #print "Looped %s number of times" % loopedTimes
            activeColumns = activeColumnVect.reshape((self.height, self.width))
            #print "activeColumns = \n%s" % activeColumns
            #print "original overlaps = \n%s" % overlapsGrid

        return activeColumnVect

    def sortOverlapMatrix(self, colOverlapVals):
        # colOverlapVals, each row is a list of overlaps values that
        # a column can potentially inhibit.
        # Sort the grid of overlap values from largest to
        # smallest for each columns inhibit overlap vect.
        sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
        #print "sortedColOverlapsVals = \n%s" % sortedColOverlapsVals
        return sortedColOverlapsVals

    def getColInhibitionList(self, columnInd):
        # Return the input columns list of inhibition neighbours.
        # This is the list of columns that that column can inhibit.
        # The colConvolePatternIndex indicie list returned starts
        # at 1 for the first column. It also may included 0 which
        # represents padding. Need to minus one and remove all padding values.
        colIndList = self.colConvolePatternIndex[columnInd]
        colIndList = colIndList - 1
        colIndList = colIndList[colIndList >= 0]
        return colIndList


if __name__ == '__main__':

    potWidth = 2
    potHeight = 2
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 2

    # Some made up inputs to test with
    #colOverlapGrid = np.random.randint(1, size=(numRows, numCols))
    colOverlapGrid = np.array([[8, 4, 5, 8],
                               [8, 6, 1, 6],
                               [7, 7, 9, 4],
                               [2, 3, 1, 5]])
    print "colOverlapGrid = \n%s" % colOverlapGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns



