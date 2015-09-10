
import numpy as np
import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_Classes import theano_temporal

class test_temporal:
    def setUp(self):
        '''
        The theano temporal class is tested with a range of
         * potential synpase sizes
         * HTM column sizes

        '''

    def test_case1(self):
        '''
        Test the theano temporal calculator with a particular temoral case.
        '''
        potWidth = 2
        potHeight = 2
        minOverlap = 2
        numCols = 16
        timeStep = 4

        # Create an instance of the temporal calculation class
        tempPooler = theano_temporal.TemporalPoolCalculator(potWidth, potHeight, minOverlap)

        colActNotBurst = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   3, 3, 3, 3,
                                   3, 3, 3, 3])

        colOverlapVals = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   4, 4, 4, 4,
                                   0, 0, 0, 0])

        colInputPotSyn = np.array([[0, 0, 0, 0],
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

        colStopTempAtTime = np.array([0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      4, 4, 0, 0,
                                      0, 0, 0, 0])

        newTempPoolOverlapVals, updatedTempStopTime = tempPooler.calculateTemporalPool(colActNotBurst,
                                                                                       timeStep,
                                                                                       colOverlapVals,
                                                                                       colInputPotSyn,
                                                                                       colStopTempAtTime)
        colOverlapVals = colOverlapVals.reshape((4, 4))
        print "colOverlapVals = \n%s" % colOverlapVals

        colStopTempAtTime = colStopTempAtTime.reshape((4, 4))
        print "colStopTempAtTime = \n%s" % colStopTempAtTime

        newTempPoolOverlapVals = newTempPoolOverlapVals.reshape((4, 4))
        print "newTempPoolOverlapVals = \n%s" % newTempPoolOverlapVals
        updatedTempStopTime = updatedTempStopTime.reshape((4, 4))
        print "updatedTempStopTime = \n%s" % updatedTempStopTime

        # assert len(newTempPoolOverlapVals) == numCols

    def test_case2(self):
        '''
        Test the theano temporal calculator with a second particular temoral case.
        '''
        potWidth = 2
        potHeight = 2
        minOverlap = 2
        numCols = 16
        timeStep = 4

        # Create an instance of the temporal calculation class
        tempPooler = theano_temporal.TemporalPoolCalculator(potWidth, potHeight, minOverlap)

        colActNotBurst = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   3, 3, 3, 3,
                                   3, 3, 3, 3])

        colOverlapVals = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   4, 4, 4, 4,
                                   0, 0, 0, 0])

        colInputPotSyn = np.array([[0, 0, 0, 0],
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
                                   [1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [1, 1, 0, 0]])

        colStopTempAtTime = np.array([0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      4, 4, 0, 0,
                                      0, 0, 0, 0])

        newTempPoolOverlapVals, updatedTempStopTime = tempPooler.calculateTemporalPool(colActNotBurst,
                                                                                       timeStep,
                                                                                       colOverlapVals,
                                                                                       colInputPotSyn,
                                                                                       colStopTempAtTime)
        colOverlapVals = colOverlapVals.reshape((4, 4))
        print "colOverlapVals = \n%s" % colOverlapVals

        colStopTempAtTime = colStopTempAtTime.reshape((4, 4))
        print "colStopTempAtTime = \n%s" % colStopTempAtTime

        newTempPoolOverlapVals = newTempPoolOverlapVals.reshape((4, 4))
        print "newTempPoolOverlapVals = \n%s" % newTempPoolOverlapVals
        updatedTempStopTime = updatedTempStopTime.reshape((4, 4))
        print "updatedTempStopTime = \n%s" % updatedTempStopTime

    def test_case3(self):
        '''
        Test the theano temporal calculator with a 3rd particular temoral case.
        '''
        potWidth = 2
        potHeight = 2
        minOverlap = 2
        numCols = 16
        timeStep = 6

        # Create an instance of the temporal calculation class
        tempPooler = theano_temporal.TemporalPoolCalculator(potWidth, potHeight, minOverlap)

        colActNotBurst = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   3, 3, 3, 3,
                                   4, 4, 4, 4])

        colOverlapVals = np.array([0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   4, 4, 4, 4,
                                   0, 0, 0, 0])

        colInputPotSyn = np.array([[0, 0, 0, 0],
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
                                   [1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [1, 0, 0, 0]])

        colStopTempAtTime = np.array([0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      5, 5, 5, 5])

        newTempPoolOverlapVals, updatedTempStopTime = tempPooler.calculateTemporalPool(colActNotBurst,
                                                                                       timeStep,
                                                                                       colOverlapVals,
                                                                                       colInputPotSyn,
                                                                                       colStopTempAtTime)
        colOverlapVals = colOverlapVals.reshape((4, 4))
        print "colOverlapVals = \n%s" % colOverlapVals

        colStopTempAtTime = colStopTempAtTime.reshape((4, 4))
        print "colStopTempAtTime = \n%s" % colStopTempAtTime

        newTempPoolOverlapVals = newTempPoolOverlapVals.reshape((4, 4))
        print "newTempPoolOverlapVals = \n%s" % newTempPoolOverlapVals
        updatedTempStopTime = updatedTempStopTime.reshape((4, 4))
        print "updatedTempStopTime = \n%s" % updatedTempStopTime

    def test_htmSizes(self):
        '''
        Test the theano temporal calculator with a range of input sizes
        '''
        potWidth = 2
        potHeight = 2
        minOverlap = 2
        numCols = 16
        timeStep = 4

        # Create an instance of the temporal calculation class
        tempPooler = theano_temporal.TemporalPoolCalculator(potWidth, potHeight, minOverlap)

        for j in range(4, 100, 3):
            numCols = j
            print "NEW TEST ROUND"
            print "numCols = %s " % (numCols)
            colActNotBurst = np.random.randint(7, size=numCols)
            colOverlapVals = np.random.randint(potWidth * potHeight, size=(numCols))
            colInputPotSyn = np.random.randint(2, size=(numCols, potWidth * potHeight))
            colStopTempAtTime = np.random.randint(2, size=(numCols))

            newTempPoolOverlapVals, updatedTempStopTime = tempPooler.calculateTemporalPool(colActNotBurst,
                                                                                           timeStep,
                                                                                           colOverlapVals,
                                                                                           colInputPotSyn,
                                                                                           colStopTempAtTime)

            assert len(newTempPoolOverlapVals) == numCols



