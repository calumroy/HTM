import numpy as np
import math
import random

'''
A class used to update the predictive state of cells.

This class requires as inputs:
    *


def updatePredictiveState(self, timeStep):
        # The second function call for the sequence pooler.
        # Updates the predictive state of cells.
        #print "\n       2nd SEQUENCE FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                mostPredCellSynCount = 0
                # This is a count of the largest number
                #of synapses active on any segment on any cell in the column
                mostPredCell = 0
                # This is the cellIndex with the most
                # mostPredCellSynCount. This cell is the
                # highest predictor in the column.
                mostPredSegment = 0
                columnPredicting = False
                for i in range(len(c.cells)):
                    segIndex = 0
                    for s in c.cells[i].segments:
                        # This differs to the CLA.
                        # When all cells are active in a
                        # column this stops them from all causing predictions.
                        # lcchosen will be correctly set when a
                        # cell predicts and is activated by a group of learning cells.
                        #activeState = 1
                        #if self.segmentActive(s,timeStep,activeState) > 0:
                        #learnState = 2
                        # Use active state since a segment sets a cell into the predictive
                        # state when it contains many synapses connected to currently active cells.
                        activeState = 1
                        predictionLevel = self.segmentActive(s, timeStep, activeState)
                        #if predictionLevel > 0:
                        #    print "x,y,cell = %s,%s,%s predLevel =
                        #%s"%(c.pos_x,c.pos_y,i,predictionLevel)
                        # Check that this cell is the highest
                        #predictor so far for the column.
                        if predictionLevel > mostPredCellSynCount:
                            mostPredCellSynCount = predictionLevel
                            mostPredCell = i
                            mostPredSegment = segIndex
                            columnPredicting = True
                        segIndex = segIndex+1
                        # Need this to hand to getSegmentActiveSynapses\
                if columnPredicting is True:
                    # Set the most predicting cell in
                    # the column as the predicting cell.
                    self.predictiveStateAdd(c, mostPredCell, timeStep)
                    # Only create a new update structure if the cell wasn't already predicting
                    if self.predictiveState(c, mostPredCell, timeStep-1) is False:
                        activeUpdate = self.getSegmentActiveSynapses(c, mostPredCell, timeStep, mostPredSegment, False)
                        c.cells[mostPredCell].segmentUpdateList.append(activeUpdate)

'''


class predictiveCells():
    def __init__(self, numColumns, cellsPerColumn, numSegPerCell, activationThreshold):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        self.numSegPerCell = numSegPerCell
        self.activationThreshold = activationThreshold
        # The predictive cells. This is a 2D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each cell stores the timestep that the cell was last in the predictive state.
        self.predictCells = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # activeSeg is a 3D tensor. The first dimension is the columns, the second the cells and the 3rd is
        # the segment in the cells. For each segment a timeStep is stored indicating when the segment was
        # last in an active state. This means it was predicting that the cell would become active in the
        # next timeStep. This is what the CLA paper calls a "SEQUENCE SEGMENT".
        self.activeSeg = np.array([[[-1 for z in range(self.numSegPerCell)]
                                   for x in range(self.cellsPerColumn)]
                                   for y in range(self.numColumns)])

    def segmentActive(self, segmentSynList, activeCells, timeStep):
        # In the segments list of synapses check if the number of
        # synapses connected to active cells is larger then
        # the self.activationThreshold. If so return the number
        # of synpases with the state.
        # The segment Synapse list is an input 2 D tensor where the
        # first dimension stores a list of synapses with each synapse
        # storing [columnIndex, cellIndex, permanence]. The column index
        # and cell index store the location that the synapse connects to.
        count = 0
        for i in range(len(segmentSynList)):
            # Only check synapses that have a large enough permanence
            synPermanence = segmentSynList[i][2]
            if synPermanence > self.connectPermanence:
                colIndex = segmentSynList[i][0]
                cellIndex = segmentSynList[i][0]
                if activeCells[colIndex][cellIndex] == timeStep:
                    count += 1
        if count > self.activationThreshold:
            # print"         %s synapses were active on segment"%count
            return count
        else:
            return 0

    def updatePredictState(self, activeCells, distalSynapses, timeStep):
        '''
        Updates the predictive state of cells.

        Inputs
            1. activeCells is a 2D tensor. The first dimension stores the columns the second is the cells
               in the columns. Each cell stores the timeStep of when that cell was last in the active state.

            2. distalSynapses is a 5D tensor. The first dimension stores the columns, the 2nd is the cells
               in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
               segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
               This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
               It does not change size. Its size is fixed when this class is constructed.

        Outputs
            1. self.predictCells a 2d array holding the timeStep of when each
               cell was last in a predicitve state.
        '''

        for c in range(len(distalSynapses)):
            # This is a count of the largest number
            # of synapses active on any segment on any cell in the column
            mostPredCellSynCount = 0
            # This is the cellIndex with the most
            # mostPredCellSynCount. This cell is the
            # highest predictor in the column.
            mostPredCell = 0
            mostPredSegment = 0
            columnPredicting = False
            # Iterate through each cell in the column
            for i in range(len(distalSynapses[c])):
                # Iterate through each segment in the cell
                for s in range(len(distalSynapses[c][i])):
                    segmentSynList = distalSynapses[c][i][s]
                    predictionLevel = self.segmentActive(segmentSynList, activeCells, timeStep)
                    # If the segment is active then this means the segment is predicting
                    # Store this activity in the activeSeg 3D tensor.
                    if predictionLevel > 0:
                        self.activeSeg[c][i][s] = timeStep
                    # Check if this segment is the most active for the cell.
                    if predictionLevel > mostPredCellSynCount:
                            mostPredCellSynCount = predictionLevel
                            mostPredCell = i
                            mostPredSegment = s
                            columnPredicting = True
            if columnPredicting is True:
                # Only create a new update structure if the cell chosen as
                # the most predicting wasn't already predicting
                if self.predictCells[c][mostPredCell] != timeStep-1:
                    activeUpdate = self.getSegmentActiveSynapses(c, mostPredCell, timeStep, mostPredSegment, False)
                    c.cells[mostPredCell].segmentUpdateList.append(activeUpdate)
                # Set the most predicting cell in the column as the predicting cell.
                self.predictCells[c][mostPredCell] = timeStep


if __name__ == '__main__':
    numRows = 4
    numCols = 4
    cellsPerColumn = 3
    numColumns = numRows * numCols
