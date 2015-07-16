import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math

'''
A class to calculate the temporal pooling for a HTM layer.
This class uses theano functions to speed up the computation.
It can be implemented on a GPU see theano documents for
enabling GPU calculations.

Inputs:
It uses the overlap values for each column and an matrix
of values specifying when a column was active but not bursting last.

Outputs:
It outputs a matrix of new overlap values for each column where
the columns that are temporally pooling are given a maximum overlap value.
'''


class TemporalPoolCalculator():
    def __init__(self, potentialWidth, potentialHeight,
                 centerPotSynapses, connectedPerm,
                 minOverlap):
        # Temporal Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerPotSynapses = centerPotSynapses
        self.potentialWidth = potentialWidth
        self.potentialHeight = potentialHeight
        self.connectedPermParam = connectedPerm
        self.minOverlap = minOverlap

        # Create theano variables and functions
        ############################################
