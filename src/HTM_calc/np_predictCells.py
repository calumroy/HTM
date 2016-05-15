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


class predictCellsCalculator():
    def __init__(self, numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, connectPermanence):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        # Maximum number of segments per cell
        self.maxSegPerCell = maxSegPerCell
        # Maximum number of synapses per segment
        self.maxSynPerSeg = maxSynPerSeg
        # The minimum required permanence value required by a synapse for it
        # to be connected.
        self.connectPermanence = connectPermanence
        # A tensor storing for each cells segments the number of synapses connected to active cells.
        self.currentSegSynCount = np.zeros((self.numColumns,
                                            self.cellsPerColumn,
                                            maxSegPerCell))
        # The timeSteps when cells where in the predicitive state last. This is a 3D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timestep when this cell was predicitng last.
        self.predictCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdate = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        self.segActiveSyn = np.array([[[-1 for z in range(self.maxSynPerSeg)]
                                     for x in range(self.cellsPerColumn)]
                                     for y in range(self.numColumns)])

    def getSegUpdates(self):
        # Return the tensors storing information on which distal synapses
        # in which segments learning should be performed on.
        return self.segIndUpdate, self.segActiveSyn

    def getSegmentActiveSynapses(self, segSynapseList, timeStep, activeCellsTime):
        # Find the synapse indicies of the active synapses for the timestep.
        # Synapses whose end is on an active cell for the timestep.
        # The segSynapseList stores an array of synapses where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        # Return an array of size len(segSynapseList) where a 1 indicates
        # the synapse is active and a 0 indicates inactive.
        activeSynapses = np.zeros(len(segSynapseList))
        for i in range(len(segSynapseList)):
            columnIndex = segSynapseList[i][0]
            cellIndex = segSynapseList[i][1]
            if segSynapseList[i][2] >= self.connectPermanence:
                if self.checkCellActive(columnIndex, cellIndex, timeStep, activeCellsTime) is True:
                    activeSynapses[i] = 1
                else:
                    activeSynapses[i] = 0
            else:
                activeSynapses[i] = 0

        return activeSynapses

    def checkCellPredicting(self, colIndex, cellIndex, timeStep):
        # Check if the given cell was in the predicitng state at the timestep given.
        # We check the 3D predictingCellsTensor tensor which holds multiple
        # previous timeSteps when each cell was last in the predicitng state.
        if self.predictCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if self.predictCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def setPredictCell(self, colIndex, cellIndex, timeStep):
        # Set the given cell at colIndex, cellIndex into a predictive state for the
        # given timeStep.
        # We need to check the predictCellsTime tensor which holds multiple
        # previous timeSteps and set the oldest one to the given timeStep.
        if self.predictCellsTime[colIndex][cellIndex][0] <= self.predictCellsTime[colIndex][cellIndex][1]:
            self.predictCellsTime[colIndex][cellIndex][0] = timeStep
        else:
            self.predictCellsTime[colIndex][cellIndex][1] = timeStep

    def checkCellActive(self, colIndex, cellIndex, timeStep, activeCellsTime):
        # Check if the given cell was active at the timestep given.
        # We need to check the activeCellsTime tensor which holds multiple
        # previous timeSteps when each cell was last active.
        # import ipdb; ipdb.set_trace()
        if activeCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if activeCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def segmentNumSynapsesActive(self, synapseMatrix, timeStep, activeCellsTime):
        # Find the number of active synapses for the current timestep.
        # Synapses whose end is on an active cell for the current timestep.
        # The synapseMatrix stores an array of synpases where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        # Also make sure the synapse has a permanence larger then the minimum
        # connected value.
        count = 0
        for i in range(len(synapseMatrix)):
            if synapseMatrix[i][2] > self.connectPermanence:
                columnIndex = int(synapseMatrix[i][0])
                cellIndex = int(synapseMatrix[i][1])
                if self.checkCellActive(columnIndex, cellIndex, timeStep, activeCellsTime) == True:
                    count += 1

        return count

    def updatePredictiveState(self, timeStep, activeCellsTime, distalSynapses):
    	'''
        This function calculates which cells should be set into the predictive state.
        '''

        for c in range(self.numColumns):
            mostPredCellSynCount = 0
            mostPredCell = 0
            columnPredicting = False

            for i in range(self.cellsPerColumn):
                for g in range(self.maxSegPerCell):
                    # Find in the current segment, synapses that where active for the current timeStep.
                    predictionLevel = self.segmentNumSynapsesActive(distalSynapses[c][i][g],
                                                                    timeStep,
                                                                    activeCellsTime)

                    self.currentSegSynCount[c][i][g] = predictionLevel
                    if predictionLevel > mostPredCellSynCount:
                            mostPredCellSynCount = predictionLevel
                            mostPredCell = i
                            mostPredSegment = g
                            columnPredicting = True
            if columnPredicting is True:
                # Set the most predicting cell in the column as the predicting cell.
                self.setPredictCell(c, mostPredCell, timeStep)
                # Only create a new update structure if the cell wasn't already predicting
                if self.checkCellPredicting(c, i, timeStep-1) is False:
                    # Update the segment s by adding to the update tensors. The update happens in the future.
                    self.segIndUpdate[c][mostPredCell] = mostPredSegment
                    self.segActiveSyn[c][mostPredCell] = self.getSegmentActiveSynapses(distalSynapses[c][mostPredCell][mostPredSegment],
                                                                                       timeStep,
                                                                                       activeCellsTime)
                    print "self.segActiveSyn[%s][%s] = %s" % (c, mostPredCell, self.segActiveSyn[c][mostPredCell])

        print "self.currentSegSynCount = \n%s" % self.currentSegSynCount
        #print "self.predictCellsTime = \n%s" % self.predictCellsTime

        return self.predictCellsTime


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
    timeStep = 1

    # Create the distalSynapse 5d tensor holding the information of the distal synapses.
    distalSynapses = np.zeros((numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, 3))
    for index, x in np.ndenumerate(distalSynapses):
        # print index, x
        if index[4] == 2:
            distalSynapses[index] = random.randint(0, 10) / 10.0
        if index[4] == 1:
            distalSynapses[index] = random.randint(0, cellsPerColumn-1)
        if index[4] == 0:
            distalSynapses[index] = random.randint(0, numColumns-1)

    # Create the active cells
    activeCells = np.zeros((numColumns, cellsPerColumn, 2))

    activeSeg = np.zeros((numColumns, cellsPerColumn, maxSegPerCell))

    # print "activeCells = \n%s" % activeCells
    print "distalSynapses = \n%s" % distalSynapses

    predCellsCalc = predictCellsCalculator(numColumns,
                                           cellsPerColumn,
                                           maxSegPerCell,
                                           maxSynPerSeg,
                                           connectPermanence)
    # Run through calculator

    test_iterations = 2
    for i in range(test_iterations):
        timeStep += 1
        if timeStep % 20 == 0:
            print timeStep
        print "timeStep = \n%s" % timeStep
        # Change the active columns and active cells and run again.
        activeCells = updateActiveCells(numColumns, cellsPerColumn, timeStep)
        print "activeCells = \n%s" % activeCells
        predictCellsTime = predCellsCalc.updatePredictiveState(timeStep, activeCells, distalSynapses)
        print "predictCellsTime = \n%s" % predictCellsTime
        segIndUpdate, segActiveSyn = predCellsCalc.getSegUpdates()
        print "segIndUpdate = \n%s" % (segIndUpdate)
        print "segActiveSyn = \n%s" % (segActiveSyn)


