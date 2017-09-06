import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math
import copy

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

THIS THEANO LEARNING CLASS IS A REIMPLEMENTATION OF THE np_learning.py calculator.
    
'''


class LearningCalculator():
    def __init__(self,
                 numColumns,
                 numPotSynapses,
                 spatialPermanenceInc,
                 spatialPermanenceDec,
                 activeColPermanenceDec):
        self.numColumns = numColumns
        self.numPotSynapses = numPotSynapses
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec
        # This parameter is another value used to decrement synapses permance values by.
        # It is required since already active columns decrement their synapses by a
        # different value as compared to columns that just become active.
        self.activeColPermanenceDec = activeColPermanenceDec

        # Store the previous colPotInputs.
        # This is so a potential synapse can work out if it's end
        # has changed state. If so then we update the synapses permanence.
        # Initialize with a negative value so the first update always updates
        # the permanence values. Normally this matrix holds 0 or 1 only.
        self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevActiveCols = np.array([-1 for i in range(self.numColumns)])

        # Create theano variables and functions
        ############################################
        # Create the theano function for calculating
        # the new permanence values for each columns spatial synpases.
        # A column is first checked if it is active. If so then for each
        # potential synpase inc or dec the perm value depending on if the
        # synpase was connected to an active input.
        self.col_PotInputsMat = T.matrix(dtype='int32')
        self.prev_PotInputsMat = T.matrix(dtype='int32')
        self.col_SynPermMat = T.matrix(dtype='float32')
        self.col_ActiveColVect = T.vector(dtype='int32')
        self.prev_activeColVect = T.vector(dtype='int32')
        self.row_num2 = T.matrix(dtype='int32')
        # Update the permanence values using the spatialPermanenceDec synapse decrement amount.
        # limit the permanence values between 0.0 and 1.0
        self.limit_minPerm = T.switch(T.gt(self.col_SynPermMat - self.spatialPermanenceDec, 0.0),
                                       self.col_SynPermMat - self.spatialPermanenceDec, 0.0)
        self.limit_maxPerm = T.switch(T.lt(self.col_SynPermMat + self.spatialPermanenceInc, 1.0),
                                       self.col_SynPermMat + self.spatialPermanenceInc, 1.0)
        self.check_inputAct = T.switch(T.gt(self.col_PotInputsMat, 0),
                                       self.limit_maxPerm,
                                       self.limit_minPerm)
    
        # Update the permanence values using the activeColPermanenceDec instead of the normal decrement amount.
        self.limit_minPerm2 = T.switch(T.gt(self.col_SynPermMat - self.activeColPermanenceDec, 0.0),
                                       self.col_SynPermMat - self.activeColPermanenceDec, 0.0)
        self.limit_maxPerm2 = T.switch(T.lt(self.col_SynPermMat + self.spatialPermanenceInc, 1.0),
                                       self.col_SynPermMat + self.spatialPermanenceInc, 1.0)
        self.check_inputAct2 = T.switch(T.gt(self.col_PotInputsMat, 0),
                                       self.limit_maxPerm2,
                                       self.limit_minPerm2)

        
        # The column is temporally pooling or the same input is present.
        # If the synapse permanence should be decremented only
        # reduce it by the value self.activeColPermanenceDec, if any of the inputs have changed.
        self.check_potInEq = T.neq(self.col_PotInputsMat, self.prev_PotInputsMat).sum(axis=1)#T.addbroadcast(T.transpose(T.neq(self.col_PotInputsMat, self.prev_PotInputsMat).sum(axis=1)),1)
        self.check_colSameInput = T.switch(T.gt(self.check_potInEq[self.row_num2],0),
                                       self.check_inputAct2,
                                       self.col_SynPermMat)

        # Check if the column is newly activated or was active in the previous time steps.
        self.check_colIsNewAct = T.switch(T.eq(self.col_ActiveColVect[self.row_num2], self.prev_activeColVect[self.row_num2]),
                                       self.check_colSameInput,
                                       self.check_inputAct)

        #Check that the column is active before updating it's permanence.
        self.check_colIsActive = T.switch(T.gt(self.col_ActiveColVect[self.row_num2], 0),
                                          self.check_colIsNewAct, 
                                          self.col_SynPermMat)
        #self.check_colIsActive = self.check_colSameInput

        #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
        self.get_updateSynPerm = function([self.col_SynPermMat,
                                           self.col_PotInputsMat,
                                           self.col_ActiveColVect,
                                           self.row_num2,
                                           self.prev_PotInputsMat,
                                           self.prev_activeColVect],
                                          self.check_colIsActive,
                                          on_unused_input='ignore',
                                          allow_input_downcast=True
                                          )

        # #TODO remove this
        # self.col_PotInputsMat2 = T.matrix(dtype='int32')
        # self.prev_PotInputsMat2 = T.matrix(dtype='int32')
        # self.check_potInEq2 = T.neq(self.col_PotInputsMat2, self.prev_PotInputsMat2)#.sum(axis=1)
        # self.get_updateSynPerm2 = function([self.col_PotInputsMat2,
        #                                    self.prev_PotInputsMat2],
        #                                   self.check_potInEq2,
        #                                   on_unused_input='ignore',
        #                                   allow_input_downcast=True
        #                                   )

        #### END of Theano functions and variables definitions
        #################################################################
        # Create a matrix that just holds the column number for each element
        self.row_num = np.array([[j for i in range(self.numPotSynapses)]
                                for j in range(self.numColumns)])

    def updatePermanenceValues(self, colSynPerm, colPotInputs, activeCols):
        
        #print "self.row_num = \n%s" %self.row_num
        newPermanceMat = self.get_updateSynPerm(colSynPerm,
                                                colPotInputs,
                                                activeCols,
                                                self.row_num,
                                                self.prevColPotInputs,
                                                self.prevActiveCols
                                                )

        # Store the current inputs to the potentialSynapses to use next time.
        self.prevColPotInputs = colPotInputs
        self.prevActiveCols = activeCols

        return newPermanceMat


if __name__ == '__main__':

    numRows = 4
    numCols = 4
    spatialPermanenceInc = 0.01
    spatialPermanenceDec = 0.01
    maxNumTempoPoolPatterns = 3
    activeColPermanenceDec = 0.4 #float(spatialPermanenceInc)/float(maxNumTempoPoolPatterns)
    numPotSyn = 4
    numColumns = numRows * numCols
    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # Create an array representing the potential inputs to each column
    colPotInputsMat = np.random.randint(2, size=(numColumns, numPotSyn))
    # Create an array representing the active columns
    activeCols = np.random.randint(2, size=(numColumns))

    print "colSynPerm = \n%s" % colSynPerm
    print "colPotInputsMat = \n%s" % colPotInputsMat
    print "activeCols = \n%s" % activeCols

    permanenceUpdater = LearningCalculator(numColumns,
                                           numPotSyn,
                                           spatialPermanenceInc,
                                           spatialPermanenceDec,
                                           activeColPermanenceDec)

    colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                          colPotInputsMat,
                                                          activeCols)

    print "UPDATED colSynPerm = \n%s" % colSynPerm

    print "SECOND STEP"

    # Changed the first element in the colPotInputsMat.
    # Need to copy the matrix otherwise the previous value stored in the class is also updated.
    colPotInputsMat = copy.copy(colPotInputsMat)
    for i in range(len(colPotInputsMat[0])):
        if colPotInputsMat[0][i] == 0:
            colPotInputsMat[0][i] = 1
        else:
            colPotInputsMat[0][i] = 0

    print "colSynPerm = \n%s" % colSynPerm
    print "colPotInputsMat = \n%s" % colPotInputsMat
    print "activeCols = \n%s" % activeCols

    colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                          colPotInputsMat,
                                                          activeCols)

    print "UPDATED colSynPerm = \n%s" % colSynPerm



