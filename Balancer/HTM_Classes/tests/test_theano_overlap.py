
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_Classes import theano_overlap


class test_theanoOverlap:
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

    def test_minOverlap(self):
        '''
        Test the theano overlap calculator with a case where their is no
        columns with an overlap value larger then the min overlap value.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 1
        numColumnRows = 4
        numColumnCols = 4
        connectedPerm = 0.3
        minOverlap = 3

        colSynPerm = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])

        newInputMat = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 0],
                                [1, 1, 1, 1],
                                [0, 0, 0, 0]])

        numInputCols = 4
        numInputRows = 4

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

        # Return both the overlap values and the inputs from
        # the potential synapses to all columns.
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

        # limit the overlap values so they are larger then minOverlap
        colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

        #import ipdb; ipdb.set_trace()
        assert np.sum(colOverlaps) == 0

    def test_uncenteredCase1(self):
        '''
        Test the theano overlap calculator with a case where
        each column calculates the overlap with that columns
        potential synpases begining from the top right. The
        potential synpases are not cenetered around the column.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 0
        numColumnRows = 5
        numColumnCols = 4
        connectedPerm = 0.3
        minOverlap = 3

         # The below colsynPerm needs to have potWidth * potHeight number of columns
         # and needs to have numColumnCols * numColumnRows number of rows.
        colSynPerm = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])

        # Needs to have numColumnCols number of columns and
        # numColumnRows number of rows for it to be valid.
        newInputMat = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 0],
                                [1, 1, 1, 1],
                                [0, 0, 0, 0]])

        numInputCols = 4
        numInputRows = 4

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

        # Return both the overlap values and the inputs from
        # the potential synapses to all columns.
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

        # limit the overlap values so they are larger then minOverlap
        colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

        #import ipdb; ipdb.set_trace()
        assert np.sum(colOverlaps) == 0

    def test_uncenteredInputSizes(self):
        potWidth = 4
        potHeight = 4
        centerPotSynapses = 0
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


