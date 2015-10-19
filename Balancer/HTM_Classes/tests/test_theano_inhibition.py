
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_Classes import theano_inhibition

class test_theanoInhibition:
    def setUp(self):
        '''
        The theano inhibition class is tested with a range of
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

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        #print "activeColumns = \n%s" % activeColumns

        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_case2(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
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

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[1, 0, 1, 1],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [1, 0, 0, 1]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_case3(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 2
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 4
        numCols = 4
        desiredLocalActivity = 2

        colOverlapGrid = np.array([[8, 4, 5, 8],
                                   [8, 6, 1, 6],
                                   [7, 7, 9, 4],
                                   [2, 3, 1, 5]])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[1, 0, 1, 1],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [1, 1, 0, 1]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_largeInput(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 10
        inhibitionHeight = 10
        centerInhib = 1
        numRows = 100
        numCols = 200
        desiredLocalActivity = 2

        # Some made up inputs to test with
        colOverlapGrid = np.random.randint(10, size=(numRows, numCols))

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))

        assert 1 == 1



