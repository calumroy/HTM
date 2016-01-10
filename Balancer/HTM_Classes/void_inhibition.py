import numpy as np
import math


'''
This is a class that simply activates any column with an overlap
value larger then or equal to one. It is useful when when no inhibiton is required
and execution speed is to be maximised.

Inputs:
It uses the overlap values for each column, expressed in matrix form.


Outputs:
It outputs a binary vector where each position indicates if a column
is active or not.


'''


class inhibitionCalculator():
    def __init__(self, width, height, potentialInhibWidth, potentialInhibHeight,
                 desiredLocalActivity, minOverlap, centerInhib=1):
        # This constructor method is just a placeholder so this inhibiton class looks
        # similar to the other inhibiton calculator classes.
        # We still need to calculate the neighbours lists as this is required
        # by an inhibition calculator.
        self.centerInhib = centerInhib
        self.width = width
        self.height = height
        self.potentialWidth = potentialInhibWidth
        self.potentialHeight = potentialInhibHeight
        self.minOverlap = minOverlap
        # Initialize the neighbours list for each column
        self.numColumns = self.width * self.height
        self.inhibitionArea = self.potentialWidth * self.potentialHeight
        # Calculate the neighbours list for each column.
        # Note this means we can't adjust the inhibiton width or height without
        # recalculating this list.
        self.neighbourColsLists = np.zeros((self.numColumns, self.inhibitionArea))
        self.neighbourColsLists = self.calculateNeighboursLists(self.neighbourColsLists)

    def calculateNeighboursLists(self, neighbourColsLists):
        # Returns a 2d array where each row represents a column.
        # A row contains a column index list of neighbours
        # (columns that could be inhibited) by the current column.
        # Since some columns neighbours lists are shorter then others
        # (the columns near the edges) then these lists are padded
        # with -1 index values.

        for y in range(self.height):
            for x in range(self.width):
                # Calulate the columns index number
                colIndex = x+self.width*y
                neighbourCols = self.neighbours(x, y)
                if len(neighbourCols) < self.inhibitionArea:
                    # The neighbours column needs padding
                    padNum = self.inhibitionArea - len(neighbourCols)
                    padArray = np.array([-1 for i in range(padNum)])
                    neighbourCols = np.append(neighbourCols, padArray)
                # Now set the current columns neighbours list.
                neighbourColsLists[colIndex] = neighbourCols

        #print "neighbourColsLists = \n%s" % neighbourColsLists
        return neighbourColsLists

    def neighbours(self, pos_x, pos_y):
        # returns a list of the elements that are within the inhibitionRadius
        # of the element at the given position. The returned list is a
        # list of the indicies of the elements when the overlaps grid is flattened.
        closeColumns = []
        if self.centerInhib == 0:
            # The potential inhibited columns are not centered around each column.
            # This means only the right side and bottom of the input
            # need padding.
            topPos_y = 0
            bottomPos_y = self.potentialHeight-1
            leftPos_x = 0
            rightPos_x = self.potentialWidth-1
        else:
            # The potential inhibited columns are centered over the column
            # This means all sides of the input need padding.
            topPos_y = int(math.floor(self.potentialHeight/2.0))
            bottomPos_y = int(math.ceil(self.potentialHeight/2.0))-1
            leftPos_x = int(math.floor(self.potentialWidth/2.0))
            rightPos_x = int(math.ceil(self.potentialWidth/2.0))-1

        assert topPos_y >= 0
        assert bottomPos_y >= 0
        assert leftPos_x >= 0
        assert rightPos_x >= 0

        # print "topPos_y, bottomPos_y, = %s, %s" % (topPos_y, bottomPos_y)
        # print "leftPos_x, rightPos_x = %s, %s" % (leftPos_x, rightPos_x)

        # Add one to the c.pos_y+c.inhibitionRadius because for example range(0,2)=(0,1)
        for i in range(int(pos_y-topPos_y), int(pos_y+bottomPos_y)+1):
            if i >= 0 and i < self.height:
                for j in range(int(pos_x-leftPos_x), int(pos_x+rightPos_x)+1):
                    if j >= 0 and j < self.width:
                        closeColumns.append(int(i * self.width + j))
        return np.array(closeColumns)

    def getColInhibitionList(self, columnInd):
        # Return the input columns list of inhibition neighbours.
        # This is the list of columns that that column can inhibit.
        # The self.neighbourColsLists indice list returned starts
        # at 1 for the first column. It also may included 0 which
        # represents padding. Need to minus one and remove all padding values.
        colIndList = self.neighbourColsLists[columnInd]
        colIndList = colIndList[colIndList >= 0]
        return colIndList

    def calculateWinningCols(self, overlapsGrid, potOverlapsGrid):

        allColsOverlaps = overlapsGrid.flatten().tolist()
        columnActive = np.zeros_like(allColsOverlaps)

        for i in range(len(allColsOverlaps)):
            # Make sure the overlap value is larger than or equal to one.
            overlap = allColsOverlaps[i]
            if overlap >= 1:
                columnActive[i] = 1
        # print "ACTIVE COLUMN INDICIES = \n%s" % activeColumns
        # print "columnActive = \n%s" % columnActive

        return columnActive

if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 2

    # Some made up inputs to test with
    #colOverlapGrid = np.random.randint(10, size=(numRows, numCols))
    colOverlapGrid = np.array([[8, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 7, 9, 4],
                               [0, 0, 1, 0]])
    print "colOverlapGrid = \n%s" % colOverlapGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns
