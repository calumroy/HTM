
import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_calc import theano_inhibition

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
        inhibitionWidth = 3
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 20
        numCols = 8
        desiredLocalActivity = 2
        minOverlap = 1

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
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

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
        minOverlap = 1

        colOverlapGrid = np.array([[8, 4, 5, 8],
                                   [8, 6, 1, 6],
                                   [7, 7, 9, 4],
                                   [2, 3, 1, 5]])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

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
        minOverlap = 1

        colOverlapGrid = np.array([[8, 4, 5, 8],
                                   [8, 6, 1, 6],
                                   [7, 7, 9, 4],
                                   [2, 3, 1, 5]])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

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

    def test_case4(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 2
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 6
        numCols = 5
        desiredLocalActivity = 2
        minOverlap = 1

        colOverlapGrid = np.array([[0, 0, 3, 3, 0],
                                   [0, 0, 3, 3, 0],
                                   [0, 0, 3, 3, 0],
                                   [0, 0, 3, 3, 0],
                                   [0, 0, 3, 3, 0],
                                   [0, 0, 3, 3, 0]])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0]])
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
        minOverlap = 1

        # Some made up inputs to test with
        colOverlapGrid = np.random.randint(10, size=(numRows, numCols))

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))

        assert 1 == 1

    def test_runTime(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        Run the test multiple times to get the runtime over multiple
        calls

        '''
        numCycles = 10

        inhibitionWidth = 10
        inhibitionHeight = 10
        centerInhib = 1
        numRows = 100
        numCols = 100
        desiredLocalActivity = 2
        minOverlap = 1

        # Some made up input to test with
        # We use a non random incrementing input so we can compare run times with other
        # inhibition calculator class implementations.
        colOverlapGrid = np.array([[1+i+numRows*j for i in range(numRows)] for j in range(numCols)])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity,
                                                                 minOverlap, centerInhib)

        for i in range(numCycles):
            activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))

        assert 1 == 1


