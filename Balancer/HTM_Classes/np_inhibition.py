import numpy as np
import math


'''
A class to calculate the inhibition of columns for a HTM layer.
This class uses normal numpy matrix operations.

Inputs:
It uses the overlap values for each column, expressed in matrix form.


Outputs:
It outputs a binary vector where each position indicates if a column
is active or not.

THIS IS A REINIMPLEMENTATION OF THE OLD INHIBITON CODE BELOW

    #print "length active columns before deleting = %s" % len(self.activeColumns)
    self.activeColumns = np.array([], dtype=object)
    #print "actve cols before %s" %self.activeColumns
    allColumns = self.columns.flatten().tolist()
    # Get all the columns in a 1D array then sort them based on their overlap value.
    #allColumns = allColumns[np.lexsort(allColumns.overlap, axis=None)]
    allColumns.sort(key=lambda x: x.overlap, reverse=True)
    # Now start from the columns with the highest overlap and inhibit
    # columns with smaller overlaps.
    for c in allColumns:
        if c.overlap > 0:
            # Get the neighbours of the column
            neighbourCols = self.neighbours(c)
            minLocalActivity = self.kthScore(neighbourCols, self.desiredLocalActivity)
            #print "current column = (%s, %s) overlap = %d min = %d" % (c.pos_x, c.pos_y,
            #                                                            c.overlap, minLocalActivity)
            if c.overlap > minLocalActivity:
                self.activeColumns = np.append(self.activeColumns, c)
                self.columnActiveAdd(c, timeStep)
                # print "ACTIVE COLUMN x,y = %s, %s overlap = %d min = %d" % (c.pos_x, c.pos_y,
                #                                                             c.overlap, minLocalActivity)
            elif c.overlap == minLocalActivity:
                # Check the neighbours and see how many have an overlap
                # larger then the minLocalctivity or are already active.
                # These columns will be set active.
                numActiveNeighbours = 0
                for d in neighbourCols:
                    if (d.overlap > minLocalActivity or self.columnActiveState(d, self.timeStep) is True):
                        numActiveNeighbours += 1
                # if less then the desired local activity have been set
                # or will be set as active then activate this column as well.
                if numActiveNeighbours < self.desiredLocalActivity:
                    #print "Activated column x,y = %s, %s numActiveNeighbours = %s" % (c.pos_x, c.pos_y, numActiveNeighbours)
                    self.activeColumns = np.append(self.activeColumns, c)
                    self.columnActiveAdd(c, timeStep)
                else:
                    # Set the overlap score for the losing columns to zero
                    c.overlap = 0
            else:
                # Set the overlap score for the losing columns to zero
                c.overlap = 0
        self.updateActiveDutyCycle(c)
        # Update the active duty cycle variable of every column

'''


class inhibitionCalculator():
    def __init__(self, width, height, potentialInhibWidth, potentialInhibHeight,
                 desiredLocalActivity, centerInhib=1):
        # Temporal Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerInhib = centerInhib
        self.width = width
        self.height = height
        self.potentialWidth = potentialInhibWidth
        self.potentialHeight = potentialInhibHeight
        self.numPotSyn = self.potentialWidth * self.potentialHeight
        self.desiredLocalActivity = desiredLocalActivity
        # Initialize the neighbours list for each column
        self.numColumns = self.width * self.height
        self.inhibitionArea = self.potentialWidth * self.potentialHeight
        # Calculate the neighbours list for each column.
        # Note this means we can't adjust the inhibiton width or height without
        # recalculating this list.
        self.neighbourColsLists = np.zeros((self.numColumns, self.inhibitionArea))
        self.neighbourColsLists = self.calculateNeighboursLists(self.neighbourColsLists)
        # Calculate for each column which neighbours lists they appear in.
        # Note this could be different to the neighbourColsLists if the inhibition
        # width and inhibition height are not equal.
        self.colInNeighboursLists = np.zeros((self.numColumns, self.inhibitionArea))
        self.colInNeighboursLists = self.calculateColInNeighboursLists(self.colInNeighboursLists)

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

    def calculateColInNeighboursLists(self, colInNeighboursLists):
        # Returns a 2d array where each row represents a colum index.
        # A row contains the indicies of columns that have the current
        # column in their neighbours list. E.g row one represents the column
        # with the index 0. Column 0 appears in other columns neighbours
        # lists. This means row one contains the indicies of the other columns.
        # This list can be different to a columsn neighbours list. Also
        # the lists may need to be padded with -1 values so a matrix can be formed.
        colInNeighboursLists.fill(-1)
        for y in range(self.height):
            for x in range(self.width):
                # Calulate the columns index number
                colIndex = x+self.width*y
                neighbourCols = self.neighbours(x, y)
                # Now for each of the columns in the neighbours list add
                # the current columns index to that list.
                for i in neighbourCols:
                    # Find the first -1 value in the array and replace this value.
                    last_negIndArr = np.where(colInNeighboursLists[i] == -1)[0]
                    last_negInd = last_negIndArr[0]
                    colInNeighboursLists[i][last_negInd] = colIndex

        # print "colInNeighboursLists = \n%s" % colInNeighboursLists
        return colInNeighboursLists

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

    def calculateWinningCols(self, overlapsGrid):

        activeColumns = []
        assert self.width == len(overlapsGrid[0])
        assert self.height == len(overlapsGrid)

        #print "overlapsGrid = \n%s" % overlapsGrid
        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        normValue = 1.0/float(self.numColumns+1)
        tieBreaker = np.array([[(1+i+j*self.width)*normValue for i in range(self.width)] for j in range(self.height)])
        # Add the time breaker to the overlapsGrid
        overlapsGrid = np.add(overlapsGrid, tieBreaker)

        allColsOverlaps = overlapsGrid.flatten().tolist()
        columnActive = np.zeros_like(allColsOverlaps)
        inhibitedCols = np.zeros_like(allColsOverlaps)
        # This list stores the number of active columns in a columns neighbours list.
        # It is updated and checked by other columns during the following for loop.
        numColsActInNeigh = np.zeros_like(allColsOverlaps)

        # print "columnActive = \n%s" % columnActive
        #print "overlapsGrid plus tiebreaker = \n%s" % overlapsGrid
        # print "allColsOverlaps = \n%s" % allColsOverlaps

        # Get all the columns in a 1D array then sort them based on their overlap value.
        # Return an array of the indicies containing the min overlap values to the max.
        sortedAllColsInd = np.argsort(allColsOverlaps)
        # print "sorted allColsOverlaps indicies = \n%s" % sortedAllColsInd

        # Now start from the columns with the highest overlap and inhibit
        # columns with smaller overlaps.
        for i in reversed(sortedAllColsInd):
            # Make sure the overlap value is larger than one.
            # The added tiebreaker makes a zero overlap value
            # appear in the range of 0 to 1.
            overlap = allColsOverlaps[i]
            if overlap >= 1:
                # Make sure the column hasn't been inhibited
                if inhibitedCols[i] != 1:
                    # # Get the columns position
                    # pos_x = i % self.width
                    # pos_y = math.floor(i/self.height)

                    #print "COLUMN INDEX = %s" % i

                    neighbourCols = self.neighbourColsLists[i]

                    # Check the neighbours and count how many are already active.
                    # These columns will be set active.
                    numActiveNeighbours = 0
                    for d in neighbourCols:
                        # Don't include any -1 index values. These are only padding values.
                        if d >= 0:
                            if (columnActive[d] == 1):
                                numActiveNeighbours += 1
                                # In the active neighbours see if any already have the
                                # desired number of columns active in their neighbours group.
                                # If so this column should not be set active, inhibit it.
                                if numColsActInNeigh[d] >= self.desiredLocalActivity:
                                    # print " Col index = %s has an active col with too many active neighbours" % i
                                    inhibitedCols[i] = 1

                    # Check the columns which are already active and contain the current
                    # column in its neighbours list. If one of these already has the desired
                    # local activity number of active columns then the current one should be inhibited.
                    for d in self.colInNeighboursLists[i]:
                        if (columnActive[d] == 1):
                            # Don't include any -1 index values. These are only padding values.
                            if d >= 0:
                                if numColsActInNeigh[d] >= self.desiredLocalActivity:
                                    inhibitedCols[i] = 1

                    # Store the number of active columns in this columns
                    # neighbours group. This is used and updated by other columns.
                    numColsActInNeigh[i] = numActiveNeighbours

                    # If this column wasn't inhibited and  less then the desired local
                    # activity have been set or will be set as active then activate
                    # this column as well.
                    if ((inhibitedCols[i] != 1) and (numColsActInNeigh[i] < self.desiredLocalActivity)):
                        activeColumns.append(i)
                        columnActive[i] = 1
                        # This column is activated so the numColsActInNeigh must be
                        # updated for the columns with this column in their neighbours list.
                        for c in self.colInNeighboursLists[i]:
                            # Don't include any -1 index values. These are only padding values.
                            if c >= 0:
                                # # Don't update the current column, this has already been done.
                                # if c != i:
                                # print "Adding one 100 col index = % s" % c
                                numColsActInNeigh[c] += 1
                    else:
                        # Inhibit this columns. It will not become active
                        inhibitedCols[i] = 1
                        # print " Col index = %s has too many active cols in neighbours" % i
                        # # Set the overlap score for the losing columns to zero
                        # allColsOverlaps[i] = 0
            else:
                # Inhibit this columns. It will not become active
                inhibitedCols[i] = 1
                # # Set the overlap score for the losing columns to zero
                # allColsOverlaps[i] = 0
            # print "numColsActInNeigh reshaped = \n%s" % np.array(numColsActInNeigh).reshape((self.height, self.width))
            # print "inhibitedCols reshaped = \n%s" % np.array(inhibitedCols).reshape((self.height, self.width))
            # print "allColsOverlaps reshaped = \n%s" % np.array(allColsOverlaps).reshape((self.height, self.width))
            # print "columnActive reshaped = \n%s" % np.array(columnActive).reshape((self.height, self.width))

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
    colOverlapGrid = np.array([[8, 4, 5, 8],
                               [8, 6, 1, 6],
                               [7, 7, 9, 4],
                               [2, 3, 1, 5]])
    print "colOverlapGrid = \n%s" % colOverlapGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns
