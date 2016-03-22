
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
        # The active cells. This is a 2D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the timestep when this cell was active last.
        self.activeCells = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # The predicting cells. This is a 2D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the timestep when this cell was last in the learn state.
        self.predictCells = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # The learning cells. This is a 2D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the timestep when this cell was last in the learn state.
        self.learnCells = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # The previous timesteps distal Synapse connections and permanence values. This is a 5D tensor.
        self.prevDistalSynapses = None
        # A 2d tensor storing for each columns cell a score value.
        self.cellsScore = np.array([[0 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])

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

    def getPrevActiveSegment(self, cellSegList, activeCells, activeSegList, timeStep):
        # Returns an active segment ("sequence segment") if there are none
        # then returns the most active segment
        highestActivity = 0
        mostActiveSegment = -1
        for s in range(len(cellSegList)):
            if activeSegList[s] == timeStep-1:
                # print "RETURNED SEQUENCE SEGMENT"
                return s
            else:
                segmentSynList = cellSegList[s]
                activity = self.segmentActive(segmentSynList, activeCells, timeStep-1)
                mostActiveSegment = s
                if activity > highestActivity:
                    highestActivity = activity
                    mostActiveSegment = s
        return mostActiveSegment

    def checkColPrevBursting(self, colIndex, timeStep):
        # Check that the column with the index colIndex was not bursting
        # on the previous timestep (all cells are active).
        count = 0
        for i in range(self.cellsPerColumn):
            if self.activeCells[colIndex][i] == timeStep-1:
                count += 1
            else:
                return False
        if count == self.cellsPerColumn:
            return True
        else:
            print "ERROR Column %s has %s number of cells previously active." % (colIndex, count)
            print "     A Column should have only one or all cells active!"
            return False

    def findPrevActiveCell(self, colIndex, timeStep):
        # Return the cell index that was active in the column with
        # the index colIndex for the previous timestep.
        # If all cells are active (ie bursting was occuring) then return
        # the first indicie.
        for i in range(self.cellsPerColumn):
            if self.activeCells[colIndex][i] == timeStep-1:
                return i
        # No cell was found to be active in the column.
        return None

    def findPrevLearnCell(self, colIndex, timeStep):
        # Return the cell index that was in the learn state in the column with
        # the index colIndex for the previous timestep.
        for i in range(self.cellsPerColumn):
            if self.learnCells[colIndex][i] == timeStep-1:
                return i
        # No cell was found to be in the learn state in the column.
        return None

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

    def getBestMatchingSegment(self, segmentTensor):
        # We find the segment who was most predicting for the previous timestep and call
        # this the best matching segment. This means that segment has synapses whose ends are
        # connected to cells that where active in the previous timeStep.

        # The segmentTensor is a 3d array holding all the segments for a particular cell.
        # For each segment there is an array of synpases and for each synapse there is an
        # array holding [columnIndex, cellIndex, permanence].

        # This routine is agressive. The permanence value of a synapse is allowed to be less
        # then connectedPermance but the number of active Synpses in the segment must be
        # larger then minThreshold for the segment to be considered as active.
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
                    # print "New best matching segment found for segIndex h = %s\n"%h
                    # print "Num synapses active in most active seg = %s" %(currentSegSynCount)
                    # print "Most active segIndex = %s"%(h)
            # Make sure the cell has at least one segment
            if len(segmentTensor) > 0:
                if mostActiveSegSynCount > self.minThreshold:
                    #print "Best seg found index = %s" % h
                    #print "     mostActiveSegSynCount = %s" % mostActiveSegSynCount
                    #print "     synapse list = \n%s" % segmentTensor[h]
                    return h    # returns just the index to the
                    # most active segment in the cell
        # print "returned no segment. None had enough active synapses return -1"
        return None   # Means no segment was active
        # enough and a new one will be created.

    def updateActiveCellScores(self, activeColumns, distalSynapses):
        for c in range(len(activeColumns)):
            # Only udate the scores for columns that have changed state from not active to active
            if (self.prevActiveCols[c] == 0) and (activeColumns[c] == 1):
                print "Updating score for active Column index = ", c
                # Remember the highest score in the column
                highestScore = 0
                self.colArrayHighestScoredCell[c] = -1
                # Remember the index of the cell with the highest score in the column
                for i in range(self.cellsPerColumn):
                    print "updating for cell = %s" % i
                    # Check the cell to find a best matching
                    # segment active due to active cells. Returns an index to the best segment.
                    bestMatchSeg = self.getBestMatchingSegment(distalSynapses[c][i])
                    if bestMatchSeg is not None:
                        self.cellsScore[c][i] = 1+self.segmentHighestScore(distalSynapses[c][i][bestMatchSeg])
                        # print"Cell x,y,i = %s,%s,%s bestSeg = %s score = %s"%(c.pos_x,c.pos_y,i,
                        # bestMatchSeg,c.cells[i].score)
                        if self.cellsScore[c][i] > highestScore:
                            highestScore = self.cellsScore[c][i]
                            self.colArrayHighestScoredCell[c] = i
                    else:
                        self.cellsScore[c][i] = 0

    def updateActiveCells(self, timeStep, activeColumns, predictiveCells, activeSeg, distalSynapses):

        '''
        This function calcualtes which cells should be set as active.
        It also calculates which cells should be put into a learning state.
        The learning state updates a cells synapses incrementing or decrementing the permanence value.

        Inputs:
                1.  timeStep is the number of iterations that the HTM has been through.
                    It is just an incrementing integer used to keep track of time.

                2.  activeColumns is a 1D array storing a bit indicating if the column is active (1) or not (0).

                3.  predictiveCells is a 2D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the timeStep of when that cell was last in the predictive state.

                4.  activeSeg is a 3D tensor. The first dimension is the columns, the second the cells and the 3rd is
                    the segment in the cells. For each segment a timeStep is stored indicating when the segment was
                    last in an active state. This means it was predicting that the cell would become active in the
                    next timeStep. This is what the CLA paper calls a "SEQUENCE SEGMENT".

                5.  distalSynapses is a 5D tensor. The first dimension stores the columns, the 2nd is the cells
                    in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
                    segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
                    This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
                    It does not change size. Its size is fixed when this class is constructed.

        Outputs:
                1.  activeCells is a 2d tensor storing the active state (1 active 0 not) of cells in each column.

        '''

        # Intialize the new activeCell matrix with zeros. This matrix is updated by this function.
        activeCells = np.zeros_like(self.prevActiveCells)

        # First we calculate the score for each cell in the active column
        self.updateActiveCellScores(activeColumns, distalSynapses)

        # Now we update the active and learning states of the cell.
        # This is only done for columns that are active now.
        for c in range(len(activeColumns)):
            if (activeColumns[c] == 1):
                # Any columns that are still active from the last step keep the same
                # state of cells ie. the learning and active cells stay the same.
                if (self.prevActiveCols[c] == 1):
                    if (self.checkColPrevBursting(c, timeStep) is False):
                        prevActiveCellIndex = self.findPrevActiveCell(c, timeStep)
                        self.activeCells[c][prevActiveCellIndex] = timeStep
                        self.learnCells[c][prevActiveCellIndex] = timeStep
                    else:
                        # The column bursted on the previous timestep.
                        # Leave all cells in the column active by updating the activeCells.
                        for i in range(self.cellsPerColumn):
                                self.activeCells[c][i] = timeStep
                        # Leave the previous learn cell in the learn state
                        prevLearnCellIndex = self.findPrevLearnCell(c, timeStep)
                        self.learnCells[c][prevLearnCellIndex] = timeStep
                else:
                    # For the columns that have changed state from not active to active,
                    # update their cells by setting new active and learn states.
                    # The following is very similar to the CLA paper method.
                    # These are flags indicating if an active cell and learning cell have been set yet.
                    activeCellChosen = False
                    learningCellChosen = False
                    for i in range(self.cellsPerColumn):
                        # Update the cells according to the CLA paper
                        if predictiveCells[c][i] == timeStep-1:
                            cellSegList = distalSynapses[c][i]
                            activeSegList = activeSeg[c][i]
                            s = self.getPrevActiveSegment(cellSegList, activeCells, activeSegList, timeStep)
                            # If a segment was found then continue
                            if s != -1:


        print "self.cellsScore= \n%s" % self.cellsScore
        print "self.colArrayHighestScoredCell= \n%s" % self.colArrayHighestScoredCell
        self.prevActiveCols = activeColumns
        self.prevDistalSynapses = distalSynapses


# Helper functions for the Main function.
def updateActiveCols(numColumns):
    activeColumns = np.random.randint(2, size=(numColumns))
    print "activeColumns = \n%s" % activeColumns
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
    print "activeCells = \n%s" % activeCells
    return activeCells

if __name__ == '__main__':
    # A main function to test and debug this class.
    numRows = 4
    numCols = 4
    cellsPerColumn = 3
    numColumns = numRows * numCols
    maxSegPerCell = 2
    maxSynPerSeg = 4
    minThreshold = 1

    # Create an array representing the active columns
    activeColumns = updateActiveCols(numColumns)
    ## Create an array representing the active cells.
    ##activeCells = updateActiveCells(activeColumns, cellsPerColumn)
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

    # print "activeColumns = \n%s" % activeColumns
    # print "activeCells = \n%s" % activeCells
    # print "distalSynapses = \n%s" % distalSynapses

    activeCellsCalc = activeCellsCalculator(numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, minThreshold)
    # Run through once
    activeCells = activeCellsCalc.updateActiveCells(activeColumns, distalSynapses)

    test_iterations = 1
    for i in range(test_iterations):
        # Change the active columns and active cells and run again.
        activeColumns = updateActiveCols(numColumns)
        #activeCells = updateActiveCells(activeColumns, cellsPerColumn)
        activeCells = activeCellsCalc.updateActiveCells(activeColumns, distalSynapses)



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


