import numpy as np
import math
import random

import cProfile
# Profiling function
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

'''
A class used to update the distal synapses of cells for sequence learning.

THIS CLASS IS A REIMPLEMENTATION OF THE ORIGINAL CODE:
    """
    def sequenceLearning(self, timeStep):
        # Third function called for the sequence pooler.
        # The update structures are implemented on the cells
        #print "\n       3rd SEQUENCE FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    # print "predictiveStateArray for x,y,i =
                    # %s,%s,%s is latest time = %s"%(c.pos_x,c.pos_y,i,
                        # c.predictiveStateArray[i,0])
                    if ((self.learnState(c, i, timeStep) is True) and
                        (self.learnState(c, i, timeStep-1) is False)):
                        # print "learn state for x,y,cell =
                        # %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, True)
                    # Trying a different method to the CLA white pages
                    #if self.activeState(c,i,timeStep) ==
                    #False and self.predictiveState(c,i,timeStep-1) is True:
                    if ((self.predictiveState(c, i, timeStep-1) is True and
                        self.predictiveState(c, i, timeStep) is False and
                        self.activeState(c, i, timeStep) is False)):
                        #print "INCORRECT predictive
                        #state for x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, False)
                    # After the learning delete segments if they
                    # have to few synapses or too many segments exist.
                    # This must be done after learning since during learning
                    # the index of the segment is used to identify each segment and this
                    # changes when segments are deleted.
                    self.deleteSegments(c, i)
        # Update the output of the layer
        self.updateOutput()
    """
'''


class seqLearningCalculator():
    def __init__(self, numColumns, cellsPerColumn,
                 maxSegPerCell, maxSynPerSeg, connectPermanence,
                 permanenceInc, permanenceDec):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        # Maximum number of segments per cell
        self.maxSegPerCell = maxSegPerCell
        # Maximum number of synapses per segment
        self.maxSynPerSeg = maxSynPerSeg
        # The minimum required permanence value required by a synapse for it
        # to be connected.
        self.connectPermanence = connectPermanence
        # The amount of permanence to increase a synapse by.
        self.permanenceInc = permanenceInc
        # The amount of permanence to decrease a synapse by.
        self.permanenceDec = permanenceDec

    def addNewSegSyn(self, c, i,
                     segIndNewSyn,
                     segNewSynList,
                     distalSynapses):
        # Add new synapses to the selected segment.
        if segIndNewSyn != -1:
            # print "Setting new c,i,segIndex = %s, %s, %s" % (c, i, segIndNewSyn)
            # print "     segNewSynList = \n\t%s" % segNewSynList
            for s in range(len(segNewSynList)):
                permanence = segNewSynList[s][2]
                # If the permanence value equals -1 it means don't create a new synapse, otherwise do.
                if permanence != -1:
                    # Set the synapse to the new synapses values.
                    # Set the end column, end cell and the permanence value of the new synapse.
                    distalSynapses[c][i][segIndNewSyn][s][0] = segNewSynList[s][0]
                    distalSynapses[c][i][segIndNewSyn][s][1] = segNewSynList[s][1]
                    distalSynapses[c][i][segIndNewSyn][s][2] = permanence
                    # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                    # print "     Set new syn = %s" % (segNewSynList[s])
        return distalSynapses

    def updateCurrentSegSyn(self, c, i,
                            positiveReinforcement,
                            segIndUpdate,
                            segActiveSynList,
                            distalSynapses):
        # If positive reinforcement is true then segments on the update list
        # get their permanence values increased all others get their permanence decreased.
        # If positive reinforcement is false then decrement
        # the permanence value for the active synapses.
        if segIndUpdate != -1:
            for s in range(len(segActiveSynList)):
                if positiveReinforcement is True:
                    # Increment the permanence of the active synapse
                    if segActiveSynList[s] == 1:
                        # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                        # print "Incrementing syn perm [%s,%s,%s,%s]" % (c, i, segIndUpdate, s)
                        distalSynapses[c][i][segIndUpdate][s][2] += self.permanenceInc
                        distalSynapses[c][i][segIndUpdate][s][2] = min(1.0,
                                                                       distalSynapses[c][i][segIndUpdate][s][2])
                else:
                    # Decrement the permanence of the active synapse
                    if segActiveSynList[s] == 1:
                        # print "Decrementing syn perm [%s,%s,%s,%s]" % (c, i, segIndUpdate, s)
                        distalSynapses[c][i][segIndUpdate][s][2] -= self.permanenceDec
                        distalSynapses[c][i][segIndUpdate][s][2] = max(0.0,
                                                                       distalSynapses[c][i][segIndUpdate][s][2])
                # Decrement the permanence of all synapses in the synapse list,
                # whether they were active or not.
                distalSynapses[c][i][segIndUpdate][s][2] -= self.permanenceDec
                distalSynapses[c][i][segIndUpdate][s][2] = max(0.0,
                                                               distalSynapses[c][i][segIndUpdate][s][2])
        # Return the updated distal synapses tensor.
        return distalSynapses

    def adaptSegments(self, c, i,
                      positiveReinforcement,
                      distalSynapses,
                      segIndUpdateActive,
                      segActiveSynActiveList,
                      segIndNewSynActive,
                      segNewSynActiveList,
                      segIndUpdatePredict,
                      segActiveSynPredictList):
        # Adds new segments to the cell and inc or dec the segments synapses
        # Update the synapses from the active cells update structure
        distalSynapses = self.updateCurrentSegSyn(c, i,
                                                  positiveReinforcement,
                                                  segIndUpdateActive,
                                                  segActiveSynActiveList,
                                                  distalSynapses)

        # Update the synapses from the predict cells update structure
        distalSynapses = self.updateCurrentSegSyn(c, i,
                                                  positiveReinforcement,
                                                  segIndUpdatePredict,
                                                  segActiveSynPredictList,
                                                  distalSynapses)

        # Add the new synapse to segments. These update structure are from when the cells
        # where put into the active state.
        distalSynapses = self.addNewSegSyn(c, i,
                                           segIndNewSynActive,
                                           segNewSynActiveList,
                                           distalSynapses)

    def checkCellTime(self, timeStep, colIndex, cellIndex, cellsTime):
        # Check if the given cell has the timeStep given in its history within
        # the cellsTime tensor. This could be to check that a cell was
        # active if the activeCells time tensor is given. Alternatively
        # it could be to check if the cell was in a learning or predictive state
        # if the learn or predictive cell times tensors are given instead.
        # We need to check the cellsTime tensor which holds multiple
        # previous timeSteps when each cell was last in a particular state.
        if cellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if cellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def sequenceLearning(self, timeStep, activeCellsTime,
                         learnCellsTime, predictCellsTime,
                         distalSynapses,
                         segIndUpdateActive,
                         segActiveSynActive,
                         segIndNewSynActive,
                         segNewSynActive,
                         segIndUpdatePredict,
                         segActiveSynPredict):
    	'''
        Inputs:
                1.  timeStep is the number of iterations that the HTM has been through.
                    It is just an incrementing integer used to keep track of time.

                2.  activeCellsTime is a 3D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the last two timeSteps when the cell was in an active state.
                    It must have the dimesions of self.numColumns * self.cellsPerColumn * 2.

                3.  learnCellsTime is a 3D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the last two timeSteps when the cell was in learn state.
                    It must have the dimesions of self.numColumns * self.cellsPerColumn * 2.

                4.  predictCellsTime is a 3D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the last two timeSteps when the cell was in a predictiveState.
                    It must have the dimesions of self.numColumns * self.cellsPerColumn * 2.

                5. distalSynapses is a 5D tensor. The first dimension stores the columns, the 2nd is the cells
                    in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
                    segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
                    This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
                    It does not change size. Its size is fixed when this class is constructed.

                6. Six tensors storing information on which segments to update for a cell.
                   The 6 tensors are needed because a segment can be updated by either changing permanence values of the
                   current synapses or creating new synapses or a combination of both for a single segment.
                   A cell can only store information about updating one segment at a time. There are
                   update tensors from when cells where put in the active state and from when cells where put
                   into the predictive state.
                   The six tensors are outlined below, none of them change size.

                     a. A 2D tensor "segIndUpdateActive" for each cell holds [segIndex] indicating which segment to update.
                        If the index is -1 don't update any segments.
                     b. A 3D tensor "segActiveSynActive" for each cell holds a synpase list indicating which
                        synapses in the segment (the segment index is stored in the segIndUpdateActive tensor)
                        are active [activeSynList 0 or 1].
                     c. A 2D tensor "segIndNewSynActive" for each cell holds [segIndex] indicating which segment new
                        synapses should be created for. If the index is -1 don't create any new synapses.
                     d. A 4D tensor "segNewSynActive" for each cell holds a synapse list [newSynapseListActive]
                        of new synapses that could possibly be created. Each position corresponds to a synapses
                        in the segment with the index stored in the segIndNewSyn tensor.
                        Each place in the new Synapse tensor holds [columnIndex, cellIndex, permanence]
                        If permanence is -1 then this means don't create a new synapse for that synapse.
                     a. A 2D tensor "segIndUpdatePredict" for each cell holds [segIndex] indicating which segment to update.
                        If the index is -1 don't update any segments. This is similar to the segIndUpdateActive
                     b. A 3D tensor "segActiveSynPredict" for each cell holds a synpase list indicating which
                        synapses in the segment (the segment index is stored in the segIndUpdateActive tensor)
                        are active [activeSynList 0 or 1]. This is similar to the segActiveSynActive.


        '''

        for c in range(self.numColumns):
            for i in range(self.cellsPerColumn):
                if ((self.checkCellTime(timeStep, c, i, learnCellsTime) is True) and
                   (self.checkCellTime(timeStep-1, c, i, learnCellsTime) is False)):
                    self.adaptSegments(c, i, True,
                                       distalSynapses,
                                       segIndUpdateActive[c][i],
                                       segActiveSynActive[c][i],
                                       segIndNewSynActive[c][i],
                                       segNewSynActive[c][i],
                                       segIndUpdatePredict[c][i],
                                       segActiveSynPredict[c][i])
                # Trying a different method to the CLA white pages
                if ((self.checkCellTime(timeStep-1, c, i, predictCellsTime) is True) and
                    (self.checkCellTime(timeStep, c, i, predictCellsTime) is False) and
                   (self.checkCellTime(timeStep, c, i, activeCellsTime) is False)):
                    self.adaptSegments(c, i, False,
                                       distalSynapses,
                                       segIndUpdateActive[c][i],
                                       segActiveSynActive[c][i],
                                       segIndNewSynActive[c][i],
                                       segNewSynActive[c][i],
                                       segIndUpdatePredict[c][i],
                                       segActiveSynPredict[c][i])
        return distalSynapses


# Helper functions for the Main function.
def createNewSynStructs(segIndNewSynActive, segNewSynActive,
                        numColumns, cellsPerColumn,
                        maxSegPerCell, maxSynPerSeg):

    # A 2D tensor "segIndNewSynActive" for each cell holds [segIndex] indicating which segment new
    # synapses should be created for. If the index is -1 don't create any new synapses.
    # A 4D tensor "segNewSynActive" for each cell holds a synapse list [newSynapseListActive]
    # of new synapses that could possibly be created. Each position corresponds to a synapses
    # in the segment with the index stored in the segIndNewSyn tensor.
    # Each place in the new Synapse tensor holds [columnIndex, cellIndex, permanence]
    # If permanence is -1 then this means don't create a new synapse for that synapse.

    numCells = numColumns * cellsPerColumn
    # Number of new segs made
    numNewSegs = int(0.3 * numCells)
    # Select a certain number of segment indicies to make new segments for.
    for n in range(numNewSegs):
        # Random cell chosen
        i = random.randint(0, len(segIndNewSynActive)-1)
        j = random.randint(0, len(segIndNewSynActive[0])-1)
        randSegInd = random.randint(0, maxSegPerCell-1)
        segIndNewSynActive[i][j] = randSegInd

    for index, x in np.ndenumerate(segNewSynActive):
        # print index, x
        colInd = index[0]
        cellInd = index[1]
        segInd = index[2]
        # Only set the synapse indicies to active if the corresponding segment index
        # was selected as a new segment.
        if segIndNewSynActive[colInd][cellInd] == segInd:
            if index[0] == colInd:
                if index[3] == 2:
                    segNewSynActive[index] = float(random.randint(0, 10)) / float(10.0)
                if index[3] == 1:
                    segNewSynActive[index] = random.randint(0, cellsPerColumn-1)
                if index[3] == 0:
                    segNewSynActive[index] = random.randint(0, numColumns-1)
                print "CREATED New Syn Struct for c,i,permanence = %s,%s,%s" % (int(segNewSynActive[index]), int(segNewSynActive[index]), segNewSynActive[index])

    return segIndNewSynActive, segNewSynActive


def createSynUpdateStructs(numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg):
    # Create some update distal synapse structures to test with
    segIndUpdate = np.array([[-1 for x in range(cellsPerColumn)]
                            for y in range(numColumns)])
    segActiveSyn = np.array([[[-1 for z in range(maxSynPerSeg)]
                             for x in range(cellsPerColumn)]
                             for y in range(numColumns)])

    numCells = numColumns * cellsPerColumn
    # Number of new segs made
    numNewUpdateSegs = int(0.3 * numCells)
    # Select a certain number of segment indicies to make new segments for.
    for n in range(numNewUpdateSegs):
        # Random cell chosen
        colInd = random.randint(0, len(segIndUpdate)-1)
        cellInd = random.randint(0, len(segIndUpdate[0])-1)
        randSegInd = random.randint(0, maxSegPerCell-1)
        segIndUpdate[colInd][cellInd] = randSegInd

    for index, x in np.ndenumerate(segIndUpdate):
        # print index, x
        colInd = index[0]
        cellInd = index[1]
        segInd = segIndUpdate[colInd][cellInd]
        # Chance of adding a synapses to be updated.
        chanceUpdateSyns = 0.5
        # Only set the synapse indicies to active if the corresponding segment index
        # was selected as a segment to udate.
        if segInd != -1:
            #import ipdb; ipdb.set_trace()
            for synInd in range(maxSynPerSeg):
                # Random synapse chosen
                if random.random() > chanceUpdateSyns:
                    segActiveSyn[colInd][cellInd][synInd] = 1
                    print "CREATED Update Syn Struct for c,i,segInd,synInd = %s,%s,%s,%s" % (colInd, cellInd, segInd, synInd)
                    print "     Active Synapses segActiveSyn[segInd] = %s" % segActiveSyn[colInd][cellInd]

    return segIndUpdate, segActiveSyn


def updateCellTimes(cellsTime, timeStep):
    # Set a proportion of the cells times to the current timestep
    numCells = np.prod(cellsTime.shape)
    print "numCells = %s" % numCells
    # Number of cells set
    numSetCells = int(0.3 * numCells)

    for n in range(numSetCells):
        # Random cell chosen
        i = random.randint(0, len(cellsTime)-1)
        j = random.randint(0, len(cellsTime[0])-1)
        # Rewrite over the oldest time
        prevTime1 = cellsTime[i][j][0]
        prevTime2 = cellsTime[i][j][1]
        if prevTime1 >= prevTime2:
            cellsTime[i][j][1] = timeStep
        else:
            cellsTime[i][j][0] = timeStep
    print "numSetCells = \n%s" % numSetCells
    return cellsTime


def updateActiveCellTimes(activeCellsTime, timeStep):
    # Set a proportion of the cells to active by setting the active times.
    numCells = np.prod(activeCellsTime.shape)
    print "numCells = %s" % numCells
    # Number of cells set as active
    numActiveCells = int(0.3 * numCells)

    for n in range(numActiveCells):
        # Random cell chosen
        i = random.randint(0, len(activeCellsTime)-1)
        j = random.randint(0, len(activeCellsTime[0])-1)
        # Rewrite over the oldest time
        prevTime1 = activeCellsTime[i][j][0]
        prevTime2 = activeCellsTime[i][j][1]
        if prevTime1 >= prevTime2:
            activeCellsTime[i][j][1] = timeStep
        else:
            activeCellsTime[i][j][0] = timeStep
    print "numActiveCells = \n%s" % numActiveCells
    return activeCellsTime


def createDistalSyn(distalSynapses, numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg):
    for col in range(len(distalSynapses)):
        for cell in range(len(distalSynapses[0])):
            for seg in range(len(distalSynapses[0][0])):
                for syn in range(len(distalSynapses[0][0][0])):
                    index = (col, cell, seg, syn)
                    distalSynapses[index][2] = random.randint(0, 10) / 10.0
                    distalSynapses[index][1] = random.randint(0, cellsPerColumn-1)
                    distalSynapses[index][0] = random.randint(0, numColumns-1)
    # print "distalSynapses = \n%s" % distalSynapses
    return distalSynapses


def updateActiveCells(numColumns, cellsPerColumn, timeStep):
    # Update the tensor representing the last two times each cell was active.
    # Set a random selective to active at the current timeStep

    activeCells = np.random.randint(timeStep+1, size=(numColumns, cellsPerColumn, 2))
    # print "activeColumns = \n%s" % activeColumns
    return activeCells

if __name__ == '__main__':
    # A main function to test and debug this class.
    numRows = 2
    numCols = 2
    cellsPerColumn = 2
    numColumns = numRows * numCols
    maxSegPerCell = 3
    maxSynPerSeg = 3
    connectPermanence = 0.3
    permanenceInc = 0.05
    permanenceDec = 0.02
    timeStep = 0

    # Create the distalSynapse 5d tensor holding the information of the distal synapses.
    distalSynapses = np.zeros((numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, 3))
    distalSynapses = createDistalSyn(distalSynapses, numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg)

    # Create the active cells
    activeCellsTimes = np.array([[[-1, -1] for x in range(cellsPerColumn)] for y in range(numColumns)])
    # Create the learn cell times
    learnCellsTimes = np.array([[[-1, -1] for x in range(cellsPerColumn)] for y in range(numColumns)])
    # Create the predict cell times
    predictCellsTimes = np.array([[[-1, -1] for x in range(cellsPerColumn)] for y in range(numColumns)])

    # Create the update synapse structures
    (segIndUpdateActive,
     segActiveSynActive) = createSynUpdateStructs(numColumns,
                                                  cellsPerColumn,
                                                  maxSegPerCell,
                                                  maxSynPerSeg)

    (segIndUpdatePredict,
     segActiveSynPredict) = createSynUpdateStructs(numColumns,
                                                   cellsPerColumn,
                                                   maxSegPerCell,
                                                   maxSynPerSeg)
    # Create the new synapse structure
    # Create some new distal synapse structures to test with.
    segIndNewSynActive = np.array([[-1 for x in range(cellsPerColumn)]
                                  for y in range(numColumns)])
    # Make sure the permanence is a float in the below tensor
    segNewSynActive = np.array([[[[-1, -1, -1.0] for z in range(maxSynPerSeg)]
                                for x in range(cellsPerColumn)]
                               for y in range(numColumns)])
    # Populate the structures with test data
    (segIndNewSynActive,
     segNewSynActive) = createNewSynStructs(segIndNewSynActive,
                                            segNewSynActive,
                                            numColumns,
                                            cellsPerColumn,
                                            maxSegPerCell,
                                            maxSynPerSeg)

    # print "distalSynapses = \n%s" % distalSynapses
    print "segIndUpdateActive = \n%s" % segIndUpdateActive
    print "segActiveSynActive = \n%s" % segActiveSynActive

    seqLearnCalc = seqLearningCalculator(numColumns, cellsPerColumn,
                                         maxSegPerCell, maxSynPerSeg,
                                         connectPermanence,
                                         permanenceInc, permanenceDec)

    print "STARTING SEQUENCE LEARNING CALCULATOR"
    # Run through calculator
    test_iterations = 2
    for i in range(test_iterations):
        timeStep += 1
        # if timeStep % 20 == 0:
        print "timeStep = %s" % timeStep
        activeCellsTimes = updateActiveCellTimes(activeCellsTimes, timeStep)
        predictCellsTimes = updateCellTimes(predictCellsTimes, timeStep)
        learnCellsTimes = updateCellTimes(learnCellsTimes, timeStep)
        # print "activeCellsTimes = \n%s" % activeCellsTimes
        # print "predictCellsTimes = \n%s" % predictCellsTimes
        # print "learnCellsTimes = \n%s" % learnCellsTimes

        distalSynapses = seqLearnCalc.sequenceLearning(timeStep,
                                                       activeCellsTimes,
                                                       learnCellsTimes,
                                                       predictCellsTimes,
                                                       distalSynapses,
                                                       segIndUpdateActive,
                                                       segActiveSynActive,
                                                       segIndNewSynActive,
                                                       segNewSynActive,
                                                       segIndUpdatePredict,
                                                       segActiveSynPredict)
        # print "updated distalSynapses = \n%s" % distalSynapses




