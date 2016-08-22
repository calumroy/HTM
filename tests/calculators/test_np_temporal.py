import numpy as np
# import GUI_HTM
from PyQt4 import QtGui
import sys
import json
from copy import deepcopy
from HTM_calc import np_temporal
import random


class test_np_temporal:
    def setUp(self):
        '''


        '''

    def updateLearnCellsTimes(self, timeStep, learnCellsTime, newLearnCellsList):
        # A helper function to update the Cells leanring timeStep tensor from a list of new leanring cells.
        for i in newLearnCellsList:
            colInd = i[0]
            cellInd = i[1]
            if learnCellsTime[colInd][cellInd][0] <= learnCellsTime[colInd][cellInd][1]:
                learnCellsTime[colInd][cellInd][0] = timeStep
            else:
                learnCellsTime[colInd][cellInd][1] = timeStep


    def test_case_getPrev2NewLearnCells(self):
        '''
        Test the theano temporal calculator fuction getPrev2NewLearnCells.

        This function is meant to return a list of cells that where in the learning state
        before the cells that where in the learning state in the last timestep.
        The antipenultimate learning cells.

        '''

        numRows = 2
        numCols = 4
        numColumns = numRows * numCols
        cellsPerColumn = 2
        maxSynPerSeg = 2
        spatialPermanenceInc = 1.0
        spatialPermanenceDec = 0.2
        seqPermanenceInc = 0.1
        seqPermanenceDec = 0.02
        newSynPermanence = 0.3
        minNumSynThreshold = 1
        connectPermanence = 0.2
        numPotSyn = 4
        tempDelayLength = 4
        timeStep = 5

        numCellsNeeded = maxSynPerSeg

        # Create a list storing which cells are in the learning state for the current timestep [[colInd, CellInd], ...]
        newLearnCellsList = np.array([[0, 0], [0, 1], [3, 1]])
        # Create a tensor storing the learning state cell timeSteps.
        learnCellsTime = np.zeros((numColumns, cellsPerColumn, 2))
        # Update the learning state cell timeSteps.
        self.updateLearnCellsTimes(timeStep, learnCellsTime, newLearnCellsList)

        # Create the active cells times
        activeCellsTime = np.zeros((numColumns, cellsPerColumn, 2))
        # Set some cells active in the last timestep.
        activeCellsTime[0][1][0] = timeStep-1
        activeCellsTime[1][0][1] = timeStep-1
        activeCellsTime[numColumns-1][0][0] = timeStep-1

        tempPooler = np_temporal.TemporalPoolCalculator(cellsPerColumn, numColumns, numPotSyn,
                                                        spatialPermanenceInc, spatialPermanenceDec,
                                                        seqPermanenceInc, seqPermanenceDec,
                                                        minNumSynThreshold, newSynPermanence,
                                                        connectPermanence, tempDelayLength)

        print "TimeStep =%s tempPooler.newLearnCellsTime = \n%s" % (timeStep, tempPooler.newLearnCellsTime)
        print "TimeStep =%s learnCellsTime = \n%s" % (timeStep, learnCellsTime)

        print "CALLING tempPooler.getPrev2NewLearnCells"
        potPrev2LearnCellsList = tempPooler.getPrev2NewLearnCells(timeStep, newLearnCellsList, learnCellsTime, activeCellsTime, numCellsNeeded)
        print "TimeStep =%s tempPooler.newLearnCellsTime = \n%s" % (timeStep, tempPooler.newLearnCellsTime)
        print "TimeStep =%s potPrev2LearnCellsList = \n%s" % (timeStep, potPrev2LearnCellsList)

        # Run the function again as now the tempPooler.newLearnCellsTime tensor should have been
        # updated with new cell learning state times.
        # Update the timeStep
        timeStep += 1
        # Change the new learn cells list.
        newLearnCellsList = np.array([[2, 1], [0, 1], [1, 1]])
        # Update the learning state cell timeSteps.
        self.updateLearnCellsTimes(timeStep, learnCellsTime, newLearnCellsList)
        # Set some cells active in the last timestep.
        activeCellsTime[0][1][0] = timeStep-1
        print "CALLING tempPooler.getPrev2NewLearnCells"
        potPrev2LearnCellsList = tempPooler.getPrev2NewLearnCells(timeStep, newLearnCellsList, learnCellsTime, activeCellsTime, numCellsNeeded)
        print "TimeStep =%s potPrev2LearnCellsList = \n%s" % (timeStep, potPrev2LearnCellsList)
        print "TimeStep =%s tempPooler.newLearnCellsTime = \n%s" % (timeStep, tempPooler.newLearnCellsTime)

        #assert np.array_equal(tempPooler.newLearnCellsTime, [])
