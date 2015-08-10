import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math


'''
A class used to increase or decrease the permanence
values of the potential synapses in a single HTM layer.
This class uses theano functions
to speed up the computation. Can be implemented on a GPU
see theano documents for enabling GPU calculations.

This class requires as inputs:
    * The current permanence values for each cols potential synapse.
    * A list of the connected and unconnected potential synpases for
        each column.
    * A list of the active columns.
    * How much to increment or decrement synapse values.

'''


class LearningCalculator():
    def __init__(self, spatialPermanenceInc, spatialPermanenceDec):
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec

        # Create theano variables and functions
        ############################################
        # Create the theano function for calculating
        # the new permanence values for each columns spatial synpases.
        self.col_PotInputsMat = T.matrix(dtype='int32')
        self.col_SynPermMat = T.matrix(dtype='float32')
        self.col_ActiveColVect = T.vector(dtype='int32')
        self.col_num2 = T.matrix(dtype='int32')
        self.check_inputAct = T.switch(T.gt(self.colOMat, 0), 1, 0)
        self.check_gtZero = T.switch(T.gt(self.col_ActiveColVect[self.col_num2-1], 0),
                                     self.check_inputAct, 0)
        #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
        self.get_updateSynPerm = function([self.col_SynPermMat,
                                           self.col_PotInputsMat,
                                           self.col_ActiveColVect,
                                           self.col_num2],
                                          self.check_gtZero,
                                          on_unused_input='warn',
                                          allow_input_downcast=True
                                          )

        #### END of Theano functions and variables definitions
        #################################################################
        # Create a matrix that just holds the column number for each element
        self.col_num = np.array([[i for i in range(self.potentialWidth*self.potentialHeight)]
                                for j in range(self.width*self.height)])

    def updatePermanenceValues(self, colSynPerm, colPotInputs, activeCols):
        pass


if __name__ == '__main__':

    numRows = 4
    numCols = 4
    spatialPermanenceInc = 0.1
    spatialPermanenceDec = 0.01
    numPotSyn = 9
    numColumns = numRows * numCols
    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # Create an array representing the potential inputs to each column
    colSynPerm = np.random.randint(2, size=(numRows, numCols))
    # Create an array representing the active columns
    activeCols = np.random.randint(2, size=(numColumns))

    permanenceUpdater = LearningCalculator(spatialPermanenceInc,
                                           spatialPermanenceDec)




