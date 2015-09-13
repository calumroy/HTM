
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_Classes import theano_overlap

class test_RunTime:
    def setUp(self):
        '''
        The theano overlap class is tested with a range of
         * input sizes
         * potential synpase sizes
         * HTM column sizes

        '''

    def test_inputSizes(self):
        '''
        Test the theano overlap calculator with a range of input sizes
        '''
        potWidth = 4
        potHeight = 4
        centerPotSynapses = 1
        numColumnRows = 7
        numColumnCols = 5
        connectedPerm = 0.3
        minOverlap = 3
        numPotSyn = potWidth * potHeight
        numColumns = numColumnRows * numColumnCols

        # Create an array representing the permanences of colums synapses
        colSynPerm = np.random.rand(numColumns, numPotSyn)

        for i in range(4, 100, 3):
            numInputRows = i
            for j in range(4, 100, 7):
                numInputCols = j
                print "NEW TEST ROUND"
                print "numInputRows, numInputCols = %s, %s " % (numInputRows, numInputCols)
                newInputMat = np.random.randint(2, size=(numInputRows, numInputCols))
                # Create an instance of the overlap calculation class
                overlapCalc = theano_overlap.OverlapCalculator(potWidth,
                                                               potHeight,
                                                               numColumnCols,
                                                               numColumnRows,
                                                               numInputCols,
                                                               numInputRows,
                                                               centerPotSynapses,
                                                               connectedPerm,
                                                               minOverlap)

                columnPotSynPositions = overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)

                # Return both the overlap values and the inputs from
                # the potential synapses to all columns.
                colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

                # limit the overlap values so they are larger then minOverlap
                colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

                assert len(colOverlaps) == numColumns


