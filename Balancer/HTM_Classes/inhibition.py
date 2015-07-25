import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import tensor
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
    def __init__(self, potentialInhibWidth, potentialInhibHeight,
                 desiredLocalActivity, centerInhib=1):
        # Temporal Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerInhib = centerInhib
        self.potentialWidth = potentialInhibWidth
        self.potentialHeight = potentialInhibHeight
        self.desiredLocalActivity = desiredLocalActivity

        # Create theano variables and functions
        ############################################
        # Create the theano function for calculating
        # the inputs to a column from an input grid.
        self.kernalSize = (self.potentialWidth, self.potentialHeight)
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
        # if a column should be active or not based on the sorted
        # overlap values it can inhibit and the unsorted overlap
        # inhibition values.
        # Each column has a minLocalActivity value associated with it.
        # This minLocalActivity is broadcasted from a vector to a matrix.
        # This allows comparison between each cols inhib overlap values and
        # the minLocalActivity for that col.
        self.minLocalActivity = T.matrix(dtype='float32')
        self.s_ColOMat = T.matrix(dtype='float32')

        self.check_eq_minLocAct = T.switch(T.eq(self.s_ColOMat, self.minLocalActivity),
                                           1, 0)
        self.sum_eq_matRows = self.check_eq_minLocAct.sum(axis=1)

        self.check_gt_minLocAct = T.switch(T.gt(self.s_ColOMat, self.minLocalActivity),
                                           1, 0)
        self.sum_gt_matRows = self.check_gt_minLocAct.sum(axis=1)
        self.get_colInhibActivity = function([self.s_ColOMat,
                                             self.minLocalActivity],
                                             [self.sum_gt_matRows,
                                             self.sum_eq_matRows],
                                             on_unused_input='warn',
                                             allow_input_downcast=True
                                             )

    def addPaddingToInput(self, inputGrid):
        topPos_y = 0
        bottomPos_y = 0
        leftPos_x = 0
        rightPos_x = 0

        if self.centerInhib == 0:
            # The potential inhibited columns are not centered around each column.
            # This means only the right side and bottom of the input
            # need padding.
            topPos_y = 0
            bottomPos_y = self.potentialHeight-1
            leftPos_x = 0
            rightPos_x = self.potentialWidth-1
        else:
            # The otential inhibited columns are centered over the column
            # This means all sides of the input need padding.
            topPos_y = self.potentialHeight/2
            bottomPos_y = int(math.ceil(self.potentialHeight/2.0))-1
            leftPos_x = self.potentialWidth/2
            rightPos_x = int(math.ceil(self.potentialWidth/2.0))-1

        # Make sure all are larger then zero still
        if topPos_y < 0:
            topPos_y = 0
        if bottomPos_y < 0:
            bottomPos_y = 0
        if leftPos_x < 0:
            leftPos_x = 0
        if rightPos_x < 0:
            rightPos_x = 0

        # Add the padding around the edges of the input.
        inputGrid = np.lib.pad(inputGrid,
                              ((0, 0),
                               (0, 0),
                               (topPos_y, bottomPos_y),
                               (leftPos_x, rightPos_x)),
                               'constant',
                               constant_values=(0))

        print "inputGrid = \n%s" % inputGrid

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
        print "overlapsGrid = \n%s" % overlapsGrid
        colOverlapMat = self.getColInhibInputs(overlapsGrid)
        print "colOverlapMat = \n%s" % colOverlapMat
        sortedColOverlapMat = self.sortOverlapMatrix(colOverlapMat)
        # Get the minLocalActivity for each col.
        minLocalAct = sortedColOverlapMat[:, -(self.desiredLocalActivity+1)]
        print "minLocalAct = \n%s" % minLocalAct
        # Broadcast minLocalActivity so it is the same dim as colOverlapMat
        widthColOverlapMat = len(sortedColOverlapMat[0])
        minLocalAct = np.tile(np.array([minLocalAct]).transpose(), (1, widthColOverlapMat))
        print "minLocalAct = \n%s" % minLocalAct
        # Now calculate which cols inhib overlap values are larger then and which
        # are equal to the minLocalActivity for each column.
        gtMinLocalSum, eqMinLocalSum = self.get_colInhibActivity(sortedColOverlapMat, minLocalAct)
        print "gtMinLocalSum = \n%s" % gtMinLocalSum
        print "eqMinLocalSum = \n%s" % eqMinLocalSum

    def sortOverlapMatrix(self, colOverlapVals):
        # colOverlapVals, each row is a list of overlaps values that
        # a column can potentially inhibit.
        # Sort the grid of overlap values from largest to
        # smallest for each columns inhibit overlap vect.
        sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
        print "sortedColOverlapsVals = \n%s" % sortedColOverlapsVals
        return sortedColOverlapsVals


if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 10

     # Some made up inputs to test with
    colOverlapGrid = np.random.randint(10, size=(numCols, numRows))

    inhibCalculator = inhibitionCalculator(potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    inhibCalculator.calculateInhibCols(colOverlapGrid)


