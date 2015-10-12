
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_Classes import np_inhibition

class test_npInhibition:
    def setUp(self):
        '''
        The numpy inhibition class is tested with a range of
         * potential synpase sizes
         * HTM column sizes

        '''

    def test_case1(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 2
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 20
        numCols = 8
        desiredLocalActivity = 2

        colOverlapGrid = np.array([[0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 3, 3, 0, 0, 0, 0]])

        inhibCalculator = np_inhibition.inhibitionCalculator(numCols, numRows,
                                                             inhibitionWidth, inhibitionHeight,
                                                             desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_case2(self):
        '''
        Test the np temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 3
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 4
        numCols = 4
        desiredLocalActivity = 2

        colOverlapGrid = np.array([[8, 4, 5, 8],
                                   [8, 6, 1, 6],
                                   [7, 7, 9, 4],
                                   [2, 3, 1, 5]])

        inhibCalculator = np_inhibition.inhibitionCalculator(numCols, numRows,
                                                             inhibitionWidth, inhibitionHeight,
                                                             desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[1, 0, 1, 1],
                           [1, 0, 0, 0],
                           [1, 0, 1, 0],
                           [1, 0, 0, 1]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)


