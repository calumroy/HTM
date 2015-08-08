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
    * The current permanence values.
    * A list of the connected and unconnected potential synpases for
        each column.
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
        self.col_PotInputsMat = T.matrix(dtype='float32')
        self.col_SynPermMat = T.matrix(dtype='float32')
        self.check_gt_zero = T.switch(T.gt(self.colOMat, 0), 1, 0)
        self.check_gteq_minLocAct = T.switch(T.ge(self.colOMat, self.minLocalActivity), self.check_gt_zero, 0)
        #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
        self.get_updatedSynPerm = function([self.colOMat,
                                            self.minLocalActivity],
                                           self.check_gteq_minLocAct,
                                           on_unused_input='warn',
                                           allow_input_downcast=True
                                           )

