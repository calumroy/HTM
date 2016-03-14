
import numpy as np
import math
import random


'''
A class used to activate cells

This class requires as inputs:
    *

THIS CLASS IS A REIMPLEMENTATION OF THE ORIGINAL CODE:
        """
        First function called to update the sequence pooler.
        This function has been modified to the CLA whitepaper but it resembles
        a similar modification made in NUPIC. To turn this feature off just set the
        parameter "minScoreThreshold" in the HTMLayer Class to a large number say 1000000

        It incorporates a scoring system for each cell in an active column.
        Each time a new input pattern activates a column then the cells in that column
        are given a score. The score of a cell is calculated by checking the
        getBestMatchingSegment segments synapses and locating the connected cell with the highest score.
        This score is then incremented and it becomes the current cells score.

        If the score of a cell is larger then minScoreThreshold and no predictive cell was in the
        active column then instead of bursting the column this cell becomes active.
        This active cell means it has been part of an alternative sequence that was also
        being predicted by HTM layer.
        """
        # First reset the active cells calculated from the previous time step.
        #print "       1st SEQUENCE FUNCTION"
        # This is different to CLA paper.
        # First we calculate the score for each cell in the active column
        for c in self.activeColumns:
            #print "\n ACTIVE COLUMN x,y = %s,%s time = %s"%(c.pos_x,c.pos_y,timeStep)
            # Only udate the scores for columns that have changed state from not active to active
            if self.columnActiveState(c, self.timeStep-1) is False:
                highestScore = 0        # Remember the highest score in the column
                c.highestScoredCell = None
                # Remember the index of the cell with
                #the highest score in the column
                for i in range(self.cellsPerColumn):
                    # Check the cell to find a best matching
                    #segment active due to active columns.
                    bestMatchSeg = self.getBestMatchingSegment(c, i, timeStep-1, False)
                    if bestMatchSeg != -1:
                        c.cells[i].score = 1+self.segmentHighestScore(c.cells[i].segments[bestMatchSeg], timeStep-1)
                        #print"Cell x,y,i = %s,%s,%s bestSeg = %s score = %s"%(c.pos_x,c.pos_y,i,
                        #                                                       bestMatchSeg,c.cells[i].score)
                        if c.cells[i].score > highestScore:
                            highestScore = c.cells[i].score
                            c.highestScoredCell = i
                    else:
                        c.cells[i].score = 0

        for c in self.activeColumns:
            # Only update columns that have changed state from not active to active.
            # Any columns that are still active from the last step keep the same
            # state of cells ie. the learning and active cells stay the same.
            if self.columnActiveState(c, self.timeStep-1) is True:
                prevActiveCellIndex = self.findActiveCell(c, self.timeStep-1)
                if len(prevActiveCellIndex) > 0:
                    if len(prevActiveCellIndex) == 1:
                        self.activeStateAdd(c, prevActiveCellIndex, timeStep)
                        self.learnStateAdd(c, prevActiveCellIndex, timeStep)
                        lcChosen = True
                    elif len(prevActiveCellIndex) == self.cellsPerColumn:
                        # The column bursted on the previous timestep.
                        # Leave all cells in the column active.
                        for i in range(self.cellsPerColumn):
                            self.activeStateAdd(c, i, timeStep)
                        # Leave the previous learn cell in the learn state
                        prevLearnCellIndex = self.findLearnCell(c, self.timeStep-1)
                        self.learnStateAdd(c, prevLearnCellIndex, timeStep)
                        lcChosen = True
                    else:
                        print " ERROR findActiveCell returned %s cells active for column x,y = %s,%s" % (len(prevActiveCellIndex),c.pos_x, c.posy)
                else:
                    print " ERROR column x,y = %s,%s was active but no cells are recorded as active" % (c.pos_x, c.posy)
            else:
                # According to the CLA paper
                buPredicted = False
                lcChosen = False
                for i in range(self.cellsPerColumn):
                    # Update the cells according to the CLA paper
                    if self.predictiveState(c, i, timeStep-1) is True:
                        s = self.getActiveSegment(c, i, timeStep-1)
                        # If a segment was found then continue
                        if s != -1:
                            # Since we get the active segments
                            #from 1 time step ago then we need to
                            # find which of these where sequence
                            #segments 1 time step ago. This means they
                            # were predicting that the cell would be active now.
                            if s.sequenceSegment == timeStep-1:
                                buPredicted = True
                                self.activeStateAdd(c, i, timeStep)
                                #learnState = 2
                                #if self.segmentActive(s,timeStep-1,learnState) > 0:
                                lcChosen = True
                                self.learnStateAdd(c, i, timeStep)
                # Different to CLA paper
                # If the column is about to burst because no cell was predicting
                # check the cell with the highest score.
                if c.highestScoredCell is not None:
                    if buPredicted is False and c.cells[c.highestScoredCell].score >= self.minScoreThreshold:
                        #print"best SCORE active x, y, i = %s, %s, %s score = %s"%(c.pos_x, c.pos_y,c.highestScoredCell, c.cells[c.highestScoredCell].score)
                        buPredicted = True
                        self.activeStateAdd(c, c.highestScoredCell, timeStep)
                        lcChosen = True
                        self.learnStateAdd(c, c.highestScoredCell, timeStep)
                        # Add a new Segment
                        sUpdate = self.getSegmentActiveSynapses(c, c.highestScoredCell, timeStep-1, -1, True)
                        sUpdate['sequenceSegment'] = timeStep
                        c.cells[c.highestScoredCell].segmentUpdateList.append(sUpdate)

                # According to the CLA paper
                if buPredicted is False:
                    #print "No cell in this column predicted"
                    # No prediction so the column "bursts".
                    for i in range(self.cellsPerColumn):
                        self.activeStateAdd(c, i, timeStep)
                if lcChosen is False:
                    #print "lcChosen Getting the best matching cell to set as learning cell"
                    # The best matching cell for timeStep-1
                    #is found since we want to find the
                    # cell whose segment was most active one timestep
                    #ago and hence was most predicting.
                    (cell, s) = self.getBestMatchingCell(c, timeStep-1)
                    self.learnStateAdd(c, cell, timeStep)
                    sUpdate = self.getSegmentActiveSynapses(c, cell, timeStep-1, s, True)
                    sUpdate['sequenceSegment'] = timeStep
                    c.cells[cell].segmentUpdateList.append(sUpdate)

'''


class activeCellsCalculator():
    def __init__(self, numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, minThreshold):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        # Maximum number of segments per cell
        self.maxSegPerCell = maxSegPerCell
        # Maximum number of synapses per segment
        self.maxSynPerSeg = maxSynPerSeg
        # More then this many synapses in a segment must be active for the segment
        # to be considered for an alternative sequence (to increment a cells score).
        self.minThreshold = minThreshold

        # self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevActiveCols = np.array([-1 for i in range(self.numColumns)])
        # An array storing for each column the cell index number for the cell who has the highest calculated score.
        self.colArrayHighestScoredCell = np.array([-1 for j in range(self.numColumns)])
        # The previous active cells. This is a 2D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # A 1 means the cell was previously active 0 if not.
        self.prevActiveCells = np.array([[0 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # The previous timesteps distal Synapse connections and permanence values. This is a 5D tensor.
        self.prevDistalSynapses = None
        # A 2d tensor stroing for each columns cell a score value.
        self.cellsScore = np.array([[0 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])

    def segmentHighestScore(self, segment):
        # Get the highest score of the previously active cell
        # (one timestep ago) that is connected to the end of the synapses in the segment.
        # segment is a 2d tensor, [syn1, syn2, syn3, ...] where
        # syn1 = [columnIndex, cellIndex, permanence].
        # Cells score are updated whenever they are in an
        # active column. This prevents scores getting stale.
        highestScoreCount = 0
        for i in range(len(segment)):
            columnInd = segment[i][0]
            cellInd = segment[i][1]
            if self.prevActiveCells[columnInd][cellInd] == 1:
                currentCellScore = self.cellsScore[columnInd][cellInd]
                if currentCellScore > highestScoreCount:
                    highestScoreCount = currentCellScore
        return highestScoreCount

    def segmentNumSynapsesPrevActive(self, synapseMatrix):
        # Find the number of active synapses for the previous timestep.
        # Synapses whose end is on an active cell for the previous timestep.
        # The synapseMatrix stores an array of synpases where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        count = 0
        for i in range(len(synapseMatrix)):
            columnIndex = synapseMatrix[i][0]
            cellIndex = synapseMatrix[i][1]
            if self.prevActiveCells[columnIndex][cellIndex] == 1:
                count += 1

        return count

    def getBestMatchingSegment(self, activeCells, segmentTensor):
        # This routine is agressive. The permanence value is allowed to be less
        # then connectedPermance and activationThreshold > number of active Synpses > minThreshold
        # We find the segment who was most predicting for the current timestep and call
        # this the best matching segment.
        # The input activeCells is a 2d tensor storing the active state (1 active 0 not) of cells in each column .
        # The segmentTensor is a 3d array holding all the segments for a particular cell.
        # For each segment there is an array of synpases and for each synapse there is an
        # array holding [columnIndex, cellIndex, permanence].
        # This means we need to find synapses that where previously active.
        h = 0   # mostActiveSegmentIndex
        mostActiveSegSynCount = None
        # Look through the segments for the one with the most active synapses
        # print "getBestMatchingSegment for x,y,c =
        # %s,%s,%s num segs = %s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments))
        if (self.prevDistalSynapses is not None):
            # Iterate through each segment g, for the cell i, in column c.
            for g in range(len(segmentTensor)):
                # Find in the current segment, synapses that where active for the previous timeStep.
                currentSegSynCount = self.segmentNumSynapsesPrevActive(segmentTensor[g])
                if currentSegSynCount > mostActiveSegSynCount or (mostActiveSegSynCount is None):
                    h = g
                    mostActiveSegSynCount = currentSegSynCount
                    # print "\n new best matching segment found for h = %s\n"%h
                    # print "synapses active in most active seg = %s" %(currentSegSynCount)
                    # print "Most active segIndex = %s"%(h)
            # Make sure the cell has at least one segment
            if len(segmentTensor) > 0:
                if mostActiveSegSynCount > self.minThreshold:
                    # print "returned the segment index (%s) which
                    # HAD MORE THAN THE MINTHRESHOLD SYNAPSES"%h
                    return h    # returns just the index to the
                    # most active segment in the cell
        # print "returned no segment. None had enough active synapses return -1"
        return None   # Means no segment was active
        # enough and a new one will be created.

    def updateActiveCells(self, activeColumns, activeCells, distalSynapses):

        '''
        Inputs:
                1.  activeColumns is a 1D array storing a bit indicating if the column is active (1) or not (0).

                2.  distalSynapses is a 5D tensor. The first dimmension stores the columns, the 2nd is the cells
                    in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
                    segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
                    This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
                    It does not change size. Its size is fixed when this class is constructed.
        '''

        # First we calculate the score for each cell in the active column
        for c in range(len(activeColumns)):
            # Only udate the scores for columns that have changed state from not active to active
            if (self.prevActiveCols[c] != activeColumns[c]) and (activeColumns[c] == 1):
                print "updating score for active Column index = ", c
                # Remember the highest score in the column
                highestScore = 0
                self.colArrayHighestScoredCell[c] = -1
                # Remember the index of the cell with the highest score in the column
                for i in range(self.cellsPerColumn):
                    # Check the cell to find a best matching
                    # segment active due to active cells. Returns an index to the best segment.
                    bestMatchSeg = self.getBestMatchingSegment(activeCells, distalSynapses[c][i])
                    if bestMatchSeg is not None:
                        self.cellsScore[c][i] = 1+self.segmentHighestScore(distalSynapses[c][i][bestMatchSeg])
                        # print"Cell x,y,i = %s,%s,%s bestSeg = %s score = %s"%(c.pos_x,c.pos_y,i,
                        # bestMatchSeg,c.cells[i].score)
                        if self.cellsScore[c][i] > highestScore:
                            highestScore = self.cellsScore[c][i]
                            self.colArrayHighestScoredCell[c] = i
                    else:
                        self.cellsScore[c][i] = 0

        print "self.cellsScore= \n%s" % self.cellsScore
        print "self.colArrayHighestScoredCell= \n%s" % self.colArrayHighestScoredCell

        self.prevActiveCols = activeColumns
        self.prevActiveCells = activeCells
        self.prevDistalSynapses = distalSynapses

def updateActiveCols(numColumns):
    activeColumns = np.random.randint(2, size=(numColumns))
    return activeColumns

def updateActiveCells(activeCols, cellsPerColumn):
    # note active cells are only found in active columns
    numColumns = len(activeCols)
    activeCells = np.zeros((numColumns, cellsPerColumn))
    for i in range(len(activeColumns)):
        if activeColumns[i] == 1:
            cellIndex = (random.randint(0, 3))
            # Not bursting set one of the cells active in the column.
            if cellIndex != 3:
                activeCells[i][cellIndex] = 1
            else:
                # Bursting case, set all cells active.
                activeCells[i][0:4] = 1
    return activeCells

if __name__ == '__main__':
    numRows = 4
    numCols = 4
    cellsPerColumn = 3
    numColumns = numRows * numCols
    maxSegPerCell = 2
    maxSynPerSeg = 4
    minThreshold = 2

    # Create an array representing the active columns
    activeColumns = updateActiveCols(numColumns)
    # Create an array representing the active cells.
    activeCells = updateActiveCells(activeColumns, cellsPerColumn)
    # Create the distalSynapse 5d tensor holding the information of the distal synapses.
    distalSynapses = np.zeros((numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, 3))
    for index, x in np.ndenumerate(distalSynapses):
        #print index, x
        if index[4] == 2:
            distalSynapses[index] = random.randint(0, 10) / 10.0
        if index[4] == 1:
            distalSynapses[index] = random.randint(0, cellsPerColumn-1)
        if index[4] == 0:
            distalSynapses[index] = random.randint(0, numColumns-1)

    print "activeColumns = \n%s" % activeColumns
    print "activeCells = \n%s" % activeCells
    #print "distalSynapses = \n%s" % distalSynapses

    activeCellsCalc = activeCellsCalculator(numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, minThreshold)
    # Run through once
    activeCellsCalc.updateActiveCells(activeColumns, activeCells, distalSynapses)

    test_iterations = 10
    for i in range(test_iterations):
        # Change the active columns and active cells and run again.
        activeColumns = updateActiveCols(numColumns)
        activeCells = updateActiveCells(activeColumns, cellsPerColumn)
        activeCellsCalc.updateActiveCells(activeColumns, activeCells, distalSynapses)



# def getDistalSynapses():
#     # Return this example of a distalSynapses tensor
#     distalSynapses = np.array(
#     [[[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]],



#      [[[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]],


#       [[[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]],

#        [[0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]
#         [0., 0., 0.]]]]]
#         )

#     return distalSynapses

