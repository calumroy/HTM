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

THIS THEANO LEARNING CLASS IS A REIMPLEMENTATION OF THE ORIGINAL CODE:
    for c in self.activeColumns:
        for s in c.potentialSynapses:
            # Check if the input that this
            #synapses is connected to is active.
            inputActive = self.Input[s.pos_y][s.pos_x]
            if inputActive == 1:
            #Only handles binary input sources
                s.permanence += c.spatialPermanenceInc
                s.permanence = min(1.0, s.permanence)
            else:
                s.permanence -= c.spatialPermanenceDec
                s.permanence = max(0.0, s.permanence)

    for i in range(len(self.columns)):
        for c in self.columns[i]:
            c.minDutyCycle = 0.01*self.maxDutyCycle(self.neighbours(c))
            c.updateBoost()

            if c.overlapDutyCycle < c.minDutyCycle:
                self.increasePermanence(c, 0.1*self.connectPermanence)

'''


class LearningCalculator():
    def __init__(self,
                 numColumns,
                 numPotSynapses,
                 spatialPermanenceInc,
                 spatialPermanenceDec):
        self.numColumns = numColumns
        self.numPotSynapses = numPotSynapses
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec

        # Create theano variables and functions
        ############################################
        # Create the theano function for calculating
        # the new permanence values for each columns spatial synpases.
        # A column is first checked if it is active. If so then for each
        # potential synpase inc or dec the perm value depending on if the
        # synpase was connected to an active input.
        self.col_PotInputsMat = T.matrix(dtype='int32')
        self.col_SynPermMat = T.matrix(dtype='float32')
        self.col_ActiveColVect = T.vector(dtype='int32')
        self.row_num2 = T.matrix(dtype='int32')
        self.check_inputAct = T.switch(T.gt(self.col_PotInputsMat, 0),
                                       self.col_SynPermMat + self.spatialPermanenceInc,
                                       self.col_SynPermMat - self.spatialPermanenceDec)
        self.check_colIsActive = T.switch(T.gt(self.col_ActiveColVect[self.row_num2], 0),
                                          self.check_inputAct, self.col_SynPermMat)
        #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
        self.get_updateSynPerm = function([self.col_SynPermMat,
                                           self.col_PotInputsMat,
                                           self.col_ActiveColVect,
                                           self.row_num2],
                                          self.check_colIsActive,
                                          on_unused_input='warn',
                                          allow_input_downcast=True
                                          )

        # Create the theano function for calculating
        # a limit on the synapses permanence values. They need
        # to be kept within [0.0 to 1.0].
        self.col_SynPermMat2 = T.matrix(dtype='float32')
        self.check_greaterThanOne = T.switch(T.gt(self.col_SynPermMat2, 1.0),
                                             1.0, self.col_SynPermMat2)
        self.check_lessThanZero = T.switch(T.lt(self.col_SynPermMat2, 0),
                                           0.0, self.check_greaterThanOne)
        self.limit_SynPerm = function([self.col_SynPermMat2],
                                      self.check_lessThanZero,
                                      on_unused_input='warn',
                                      allow_input_downcast=True
                                      )

        #### END of Theano functions and variables definitions
        #################################################################
        # Create a matrix that just holds the column number for each element
        self.row_num = np.array([[j for i in range(self.numPotSynapses)]
                                for j in range(self.numColumns)])

    def updatePermanenceValues(self, colSynPerm, colPotInputs, activeCols):
        newPermanceMat = self.get_updateSynPerm(colSynPerm,
                                                colPotInputs,
                                                activeCols,
                                                self.row_num
                                                )
        #print "newPermanceMat = \n%s" % newPermanceMat
        # limit the permanence values between 0.0 and 1.0
        limitedPermanceMat = self.limit_SynPerm(newPermanceMat)

        return limitedPermanceMat


# if __name__ == '__main__':

#     numRows = 4
#     numCols = 4
#     spatialPermanenceInc = 1.0
#     spatialPermanenceDec = 1.0
#     numPotSyn = 4
#     numColumns = numRows * numCols
#     # Create an array representing the permanences of colums synapses
#     colSynPerm = np.random.rand(numColumns, numPotSyn)
#     # Create an array representing the potential inputs to each column
#     colPotInputsMat = np.random.randint(2, size=(numColumns, numPotSyn))
#     # Create an array representing the active columns
#     activeCols = np.random.randint(2, size=(numColumns))

#     print "colSynPerm = \n%s" % colSynPerm
#     print "colPotInputsMat = \n%s" % colPotInputsMat
#     print "activeCols = \n%s" % activeCols

#     permanenceUpdater = LearningCalculator(numColumns,
#                                            numPotSyn,
#                                            spatialPermanenceInc,
#                                            spatialPermanenceDec)

#     colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
#                                                           colPotInputsMat,
#                                                           activeCols)


