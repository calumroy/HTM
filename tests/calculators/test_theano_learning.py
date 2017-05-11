import numpy as np
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_calc import theano_learning
from HTM_calc import np_learning

class test_theanoLearning:
    def setUp(self):
        '''
        The theano learning class is tested with a range of
         * input sizes
         * synapse permanences
         * HTM column sizes

        '''

    def test_highPerm(self):
        '''
        Test the theano learning calculator with high permanence values.
        Make sure they dont exceed 1.0
        '''
        numRows = 4
        numCols = 4
        spatialPermanenceInc = 0.3
        spatialPermanenceDec = 0.05
        activeColPermanenceDec = 0.01
        numPotSyn = 4
        numColumns = numRows * numCols
        # Create an array representing the permanences of colums synapses
        colSynPerm = np.array([[1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0]])
        # Create an array representing the potential inputs to each column
        colPotInputsMat = np.random.randint(2, size=(numColumns, numPotSyn))
        # Create an array representing the active columns
        activeCols = np.random.randint(2, size=(numColumns))

        # print "colSynPerm = \n%s" % colSynPerm
        # print "colPotInputsMat = \n%s" % colPotInputsMat
        # print "activeCols = \n%s" % activeCols

        permanenceUpdater = theano_learning.LearningCalculator(numColumns,
                                                               numPotSyn,
                                                               spatialPermanenceInc,
                                                               spatialPermanenceDec,
                                                               activeColPermanenceDec)

        colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                              colPotInputsMat,
                                                              activeCols)

        for i in range(len(colSynPerm)):
            for j in range(len(colSynPerm[i])):
                assert colSynPerm[i][j] <= 1.0
                assert colSynPerm[i][j] >= 0.0
        #print "Updated colSynPerm = \n%s" % colSynPerm

    def test_comparison(self):
        '''
        Test the theano learning calculator against the numpy learning calcualtor.
        '''
        numRows = 4
        numCols = 4
        spatialPermanenceInc = 0.5
        spatialPermanenceDec = 0.2
        activeColPermanenceDec = 0.1
        numPotSyn = 4
        numColumns = numRows * numCols
        # Create an array representing the permanences of colums synapses
        colSynPerm = np.array([[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]])
        # Create an array representing the potential inputs to each column
        colPotInputsMat = np.random.randint(2, size=(numColumns, numPotSyn))
        # Create an array representing the active columns
        activeCols = np.random.randint(2, size=(numColumns))

        # print "colSynPerm = \n%s" % colSynPerm
        # print "colPotInputsMat = \n%s" % colPotInputsMat
        # print "activeCols = \n%s" % activeCols

        permanenceUpdater = theano_learning.LearningCalculator(numColumns,
                                                               numPotSyn,
                                                               spatialPermanenceInc,
                                                               spatialPermanenceDec,
                                                               activeColPermanenceDec)

        colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                              colPotInputsMat,
                                                              activeCols)

        permanenceUpdater_np = np_learning.LearningCalculator(numColumns,
                                                               numPotSyn,
                                                               spatialPermanenceInc,
                                                               spatialPermanenceDec,
                                                               activeColPermanenceDec)

        colSynPerm_np = permanenceUpdater_np.updatePermanenceValues(colSynPerm,
                                                                    colPotInputsMat,
                                                                    activeCols)

        assert colSynPerm.all() == colSynPerm_np.all()

        # Rerun through the updatePermanenceValues
        colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                              colPotInputsMat,
                                                              activeCols)
        colSynPerm_np = permanenceUpdater_np.updatePermanenceValues(colSynPerm,
                                                                    colPotInputsMat,
                                                                    activeCols)
        assert colSynPerm.all() == colSynPerm_np.all()

        #print "Updated colSynPerm = \n%s" % colSynPerm





