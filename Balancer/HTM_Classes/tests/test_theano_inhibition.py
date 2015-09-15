
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
                                   [0, 0, 0, 0, 0, 0, 0, 0]])

        inhibCalculator = theano_inhibition.inhibitionCalculator(numCols, numRows,
                                                                 inhibitionWidth, inhibitionHeight,
                                                                 desiredLocalActivity, centerInhib)

        activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

        activeColumns = activeColumns.reshape((numRows, numCols))
        #print "activeColumns = \n%s" % activeColumns

        result = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
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
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]])
        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert activeColumns.all() == result.all()


