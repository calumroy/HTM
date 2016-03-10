
import numpy as np
import math


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
    def __init__(self, numColumns, cellsPerColumn):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn

        #self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevActiveCols = np.array([-1 for i in range(self.numColumns)])

    def getBestMatchingSegment(self, c, i, timeStep, onCell):
        # This routine is agressive. The permanence value is allowed to be less
        # then connectedPermance and activationThreshold > number of active Synpses > minThreshold
        # We find the segment who was most predicting for the current timestep and call
        # this the best matching segment.
        # This means we need to find synapses that where active at timeStep.
        # Note that this function is already called with time timeStep-1
        h = 0   # mostActiveSegmentIndex
        # Look through the segments for the one with the most active synapses
        # print "getBestMatchingSegment for x,y,c =
        # %s,%s,%s num segs = %s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments))
        for g in range(len(c.cells[i].segments)):
            # Find synapses that are active at timeStep
            currentSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[g], timeStep, onCell)
            mostActiveSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, onCell)
            if currentSegSynCount > mostActiveSegSynCount:
                h = g
                # print "\n new best matching segment found for h = %s\n"%h
                # print "segIndex = %s num of syn = %s num active syn =
                # "%(h,len(c.cells[i].segments[h].synapses),currentSegSynCount)
                # print "segIndex = %s"%(h)
        # Make sure the cell has at least one segment
        if len(c.cells[i].segments) > 0:
            if self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, onCell) > self.minThreshold:
                # print "returned the segment index (%s) which
                # HAD MORE THAN THE MINTHRESHOLD SYNAPSES"%h
                return h    # returns just the index to the
                # most active segment in the cell
        # print "returned no segment. None had enough active synapses return -1"
        return -1   # -1 means no segment was active
        # enough and a new one will be created.

    def updateActiveCells(self, activeColumns):
        # activeColumns is an array storing a bit indicating if the column is active (1) or not (0).
        # First we calculate the score for each cell in the active column
        for c in range(len(activeColumns)):
            # Only udate the scores for columns that have changed state from not active to active
            if (self.prevActiveCols[c] != activeColumns[c]) and (activeColumns[c] == 1):
                highestScore = 0        # Remember the highest score in the column
                c.highestScoredCell = None
                # Remember the index of the cell with the highest score in the column
                for i in range(self.cellsPerColumn):
                    # Check the cell to find a best matching
                    # segment active due to active cells.
                    bestMatchSeg = self.getBestMatchingSegment(c, i, timeStep-1, False)

        self.prevActiveCols = activeColumns


if __name__ == '__main__':
    numRows = 4
    numCols = 4
    cellsPerColumn = 3
    numColumns = numRows * numCols
    # Create an array representing the active columns
    activeColumns = np.random.randint(2, size=(numColumns))

    print "activeColumns = \n%s" % activeColumns

    activeCellsCalc = activeCellsCalculator(numColumns, cellsPerColumn)

    colSynPerm = activeCellsCalc.updateActiveCells(activeColumns)



