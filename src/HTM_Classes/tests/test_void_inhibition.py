
import numpy as np
# import GUI_HTM
from PyQt4 import QtGui
import sys
from HTM_Classes import void_inhibition


class test_voidInhibition:
    def setUp(self):
        '''
        The void inhibition class is tested to make sure it outputs
        valid active columns.

        '''

    def test_case1(self):
        '''
        Test the void temporal calculator with a particular temoral case.
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

        potColOverlapGrid = colOverlapGrid

        inhibCalculator = void_inhibition.inhibitionCalculator(numCols, numRows,
                                                               inhibitionWidth, inhibitionHeight,
                                                               desiredLocalActivity,
                                                               minOverlap, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid, potColOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_case2(self):
        '''
        Test the void temporal calculator with a particular temoral case.
        '''
        inhibitionWidth = 3
        inhibitionHeight = 3
        centerInhib = 1
        numRows = 4
        numCols = 4
        desiredLocalActivity = 2
        minOverlap = 1

        colOverlapGrid = np.array([[8, 0, 5, 8],
                                   [8, 0, 0, 0],
                                   [0, 0, 9, 0],
                                   [2, 0, 0, 5]])

        potColOverlapGrid = colOverlapGrid

        inhibCalculator = void_inhibition.inhibitionCalculator(numCols, numRows,
                                                               inhibitionWidth, inhibitionHeight,
                                                               desiredLocalActivity,
                                                               minOverlap, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid, potColOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        print "activeColumns = \n%s" % activeColumns

        result = np.array([[1, 0, 1, 1],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [1, 0, 0, 1]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(activeColumns, result)

    def test_runTime(self):
        '''
        Test the void inhibition calculator with a particular temoral case.
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
        potColOverlapGrid = colOverlapGrid

        inhibCalculator = void_inhibition.inhibitionCalculator(numCols, numRows,
                                                               inhibitionWidth, inhibitionHeight,
                                                               desiredLocalActivity,
                                                               minOverlap, centerInhib)

        for i in range(numCycles):
            activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid, potColOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))

        assert 1 == 1


