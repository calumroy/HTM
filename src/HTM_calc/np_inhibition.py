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
                 desiredLocalActivity, minOverlap, centerInhib=1):
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
        self.minOverlap = minOverlap
        # Initialize the neighbours list for each column
        self.numColumns = self.width * self.height
        self.inhibitionArea = self.potentialWidth * self.potentialHeight
        # Initialize the matricies storing the active and inhibited columns
        self.activeColumns = []  # This is a list storing only the active columsn indicies
        self.columnActive = None  # This is an array storing 1 (active) or 0 (inactive) for all columns.
        self.inhibitedCols = None
        # This list stores the number of active columns in a columns neighbours list.
        # It is updated and checked by other columns during the following for loop.
        self.numColsActInNeigh = None
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
        # Store the previous active columns
        # This is so a column that was previously active can be given a slight bias,
        # to break any ties that occur this timestep.
        # Initialize with a 0 no columsn where active previously.
        # the permanence values. Normally this matrix holds 0 or 1 only.
        self.prevActiveColsGrid = np.array([[0 for i in range(self.width)] for j in range(self.height)])

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

        # Add one to the c.pos_y+c.inhibition width and height because for example range(0,2)=(0,1)
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

    def addTieBreaker(self, overlapsGrid, addColBias, addBiasToPrevActive):
        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        # The tie breaker increases based on a columns index number.
        # The tie breaker is based on position and whether the column was
        # previously active or active but not bursting; the input addColBias is used to
        # indicate that extra bias should be given to a particular column that was
        # previously active. It is a matrix where each element represents a column.
        # If indicated by the input addColBias being set to 1 and addBiasToPrevActive is true then
        # add a larger bias that counts for more then the position of the column.


        #TODO
        # TRY ADDING THE TIE BREAKER TO THE ACTUAL INPUTS NOT THE COLUMN OVERLAP
        # VALUES. THIS WILL NEED TO BE DONE IN THE OVERLAP CALCULATOR.

        normValue = 1.0/float(2*self.numColumns+2)

        tieBreaker = np.array([[0 for i in range(self.width)]
                              for j in range(self.height)])
        # Create a tiebreaker that is not biased to either side of the columns grid.
        for j in range(len(tieBreaker)):
            for i in range(len(tieBreaker[0])):
                if (j % 2) == 1:
                    # For odd positions bias to the bottom left
                    tieBreaker[j][i] = ((j+1)*self.width+(self.width-i-1))*normValue
                else:
                    # For even positions bias to the bottom right
                    tieBreaker[j][i] = (1+i+j*self.width)*normValue

                #     if (i+j*self.width) % 2 == 1:
                #         # For odd positions bias to the bottom left
                #         tieBreaker[j][i] = ((j+1)*self.width+(self.width-i-1))*normValue
                #     else:
                #         # For odd positions bias to the top right
                #         tieBreaker[j][i] = ((self.height-j)*self.width+i)*normValue
                # else:
                #     if (i+j*self.width) % 2 == 1:
                #         # For even positions bias to the bottom right
                #         tieBreaker[j][i] = (1+i+j*self.width)*normValue
                #     else:
                #         # For even positions bias to the top left
                #         tieBreaker[j][i] = ((self.width-i-1)+(self.height-j)*self.width)*normValue

        #maxNormValue = (self.numColumns+1) * normValue
        # maxNormValue + normValue*numColumns must be smaller then one.
        # The maxNormValue should be larger then numColumns * normValue

        # Since the maximum tiebreaker value added to the overlap values already
        # make sure this additional value doesn't make the total more then 1.
        maxNormValue = 0.49

        # If addBiasToPrevActive is true add the tie breakers
        if addBiasToPrevActive is True:
            # It is assumed the binary grid input addColBias only holds 0 or 1.
            # We won't check for this as it is expensive.
            activeColTieBreaker = np.array([[addColBias[j][i]*maxNormValue for i in range(self.width)]
                                            for j in range(self.height)])
            # print "activeColTieBreaker=\n%s" % activeColTieBreaker
            totalTieBreaker = np.add(tieBreaker, activeColTieBreaker)
        else:
            totalTieBreaker = tieBreaker
        # Add the total tiebreaker matrix to the overlapsGrid.
        overlapsGrid = np.add(overlapsGrid, totalTieBreaker)
        #print "INHIBITION overlapsGrid + TieBreaker=\n%s" % overlapsGrid

        return overlapsGrid

    def calcualteInhibition(self, colIndex, overlapScore):
        i = int(colIndex)
        # Make sure the column hasn't been inhibited
        if self.inhibitedCols[i] != 1:
            # # Get the columns position
            # pos_x = i % self.width
            # pos_y = math.floor(i/self.height)

            neighbourCols = self.neighbourColsLists[i]

            # Check the neighbours and count how many are already active.
            # These columns will be set active.
            numActiveNeighbours = 0
            for d in neighbourCols:
                # Make sure d is an int to use as a indicie.
                d_int = int(d)
                # Don't include any -1 index values. These are only padding values.
                if d >= 0:
                    if (self.columnActive[d_int] == 1):
                        numActiveNeighbours += 1
                        # In the active neighbours see if any already have the
                        # desired number of columns active in their neighbours group.
                        # If so this column should not be set active, inhibit it.
                        if self.numColsActInNeigh[d_int] >= self.desiredLocalActivity:
                            # print " Col index = %s has an active col with too many active neighbours" % i
                            self.inhibitedCols[i] = 1

            # Check the columns which are already active and contain the current
            # column in its neighbours list. If one of these already has the desired
            # local activity number of active columns then the current one should be inhibited.
            for d in self.colInNeighboursLists[i]:
                # Make sure d is an int to use as a indicie.
                d_int = int(d)
                if (self.columnActive[d_int] == 1):
                    # Don't include any -1 index values. These are only padding values.
                    if d >= 0:
                        if self.numColsActInNeigh[d_int] >= self.desiredLocalActivity:
                            self.inhibitedCols[i] = 1

            # Store the number of active columns in this columns
            # neighbours group. This is used and updated by other columns.
            self.numColsActInNeigh[i] = numActiveNeighbours

            # If this column wasn't inhibited and  less then the desired local
            # activity have been set or will be set as active then activate
            # this column as well.
            if ((self.inhibitedCols[i] != 1) and (self.numColsActInNeigh[i] < self.desiredLocalActivity)):
                self.activeColumns.append(i)
                self.columnActive[i] = 1
                # This column is activated so the numColsActInNeigh must be
                # updated for the columns with this column in their neighbours list.
                for c in self.colInNeighboursLists[i]:
                    c_int = int(c)
                    # Don't include any -1 index values. These are only padding values.
                    if c >= 0:
                        self.numColsActInNeigh[c_int] += 1
            else:
                # Inhibit this columns. It will not become active
                self.inhibitedCols[i] = 1
                # print " Col index = %s has too many active cols in neighbours" % i
                # # Set the overlap score for the losing columns to zero
                # allColsOverlaps[i] = 0

    def calculateWinningCols(self, overlapsGrid, potOverlapsGrid):
        '''
        The Main function for this class.

        Inputs:
                1.  overlapsGrid a 2d grid (2d tensor) storing for each column the number of
                    active inputs that connected proximal (column synapses) are attached to.

                2.  potOverlapsGrid a 2d grid (2d tensor) storing for each column the number of
                    active inputs that potential proximal (column synapses) are attached to.
        Outputs:
                1.  columnActive 1d array. This is an array storing 1 (active) or 0 (inactive)
                    for all the columns.

        Function:

        Take a matrix holding all the overlap values for every column
        and calculate the active columns and the inhibitied (inactive) columns.
        To setup for the inhibiton calcualtor a tie breaker is added to all overlap values.
        This is done to resolve any situations where multiple columns have the same overlap value.
        The tie breaker takes into account a columns postion and if it was active but not bursting
        one timestep ago.

        The process of inhibitng columns is outlined below;
         1. Get the list of overlap scores for all columns.
         2. Starting from the columns with the maximum overlaps check that
            columns neighbours and decide if the column should be inhibited based
            on the number of active columns in the neighbous list and the desired
            local activity parameter.
         3. Also check any columns that are already active and also contain the
            current column in their neighbours list.
         4. If this column is to be set active then update the columns numColsActInNeigh
            variable. This keeps track of how many active columns are within that cols
            neighbours list.
         5. Also update the numColsActInNeigh parameter for any columns that contain the
            current column in their neighbours list.

        ## Potential winning columns function
        An extra feature needed from the inhibiton class is to calculate
        active columns for situations when no column has an overlap score larger
        then the minoverlap parameter.
         6. For any columns that have a overlap score less then the minoverlap parameter
            check these columns potential overlap scores and inhibit the columns that have
            a potential overlap score smaller then the minoverlap parameter.
         7. If the potential overlap score is larger then the minoverlap parameter then add a tie breaker
            value to the potential overlap score.
         8. Now rerun the inhibiton process (steps 1 to 6) this time using the list of
            potential overlap scores.


        '''

        self.activeColumns = []
        assert self.width == len(overlapsGrid[0])
        assert self.height == len(overlapsGrid)

        # Add a tie breaker to the overlapsGrid based on position and
        # if the column was previously active.
        #biasPrevActiveCols = False
        #overlapsGrid = self.addTieBreaker(overlapsGrid, self.prevActiveColsGrid, biasPrevActiveCols)
        # Do the same for the potential Overlaps Grid.
        #biasPrevActiveCols = False
        #potOverlapsGrid = self.addTieBreaker(potOverlapsGrid, self.prevActiveColsGrid, biasPrevActiveCols)

        allColsOverlaps = overlapsGrid.flatten().tolist()
        allColsPotOverlaps = potOverlapsGrid.flatten().tolist()

        self.columnActive = np.zeros_like(allColsOverlaps)
        self.inhibitedCols = np.zeros_like(allColsOverlaps)
        # This list stores the number of active columns in a columns neighbours list.
        # It is updated and checked by other columns during the following for loop.
        self.numColsActInNeigh = np.zeros_like(allColsOverlaps)

        # print "columnActive = \n%s" % columnActive
        # print "overlapsGrid plus tiebreaker = \n%s" % overlapsGrid
        # print "allColsOverlaps = \n%s" % allColsOverlaps

        # Get all the columns in a 1D array then sort them based on their overlap value.
        # Return an array of the indicies containing the min overlap values to the max.
        sortedAllColsInd = np.argsort(allColsOverlaps)
        # print "sorted allColsOverlaps indicies = \n%s" % sortedAllColsInd

        # Now start from the columns with the highest overlap and inhibit
        # columns with smaller overlaps.
        for i in reversed(sortedAllColsInd):
            # Make sure the overlap value is larger than the minOverlap.
            # The added tiebreaker makes a zero overlap value
            # appear in the range of 0 to 1.
            overlap = allColsOverlaps[i]
            if overlap >= self.minOverlap:
                self.calcualteInhibition(i, overlap)
            else:
                # The remaining columns all have too small overlap values.
                break

        # The remaining columns may still become active if they have good potential overlap values
        # and they haven't been inhibited by the already active columns.
        # Start from the columns with the highest potential overlap values.
        sortedAllColsPotInd = np.argsort(allColsPotOverlaps)
        for i in reversed(sortedAllColsPotInd):
            # Don't look at any columns that have already been activated or inhibited.
            if self.inhibitedCols[i] == 0 and self.columnActive[i] == 0:
                # Check the columns potential overlap score.
                potOverlap = allColsPotOverlaps[i]
                if potOverlap >= self.minOverlap:
                    # Check that this column is not inhibited by any active column,
                    # if not then set active.
                    self.calcualteInhibition(i, potOverlap)

            # print "numColsActInNeigh reshaped = \n%s" % np.array(numColsActInNeigh).reshape((self.height, self.width))
            #print "inhibitedCols reshaped = \n%s" % np.array(self.inhibitedCols).reshape((self.height, self.width))
            # print "allColsOverlaps reshaped = \n%s" % np.array(allColsOverlaps).reshape((self.height, self.width))
            #print "columnActive reshaped = \n%s" % np.array(self.columnActive).reshape((self.height, self.width))

        # print "ACTIVE COLUMN INDICIES = \n%s" % activeColumns
        # print "columnActive = \n%s" % columnActive

        # Save the columActive array so the prev active columns
        # are known and can be used next time this function is called.
        self.prevActiveColsGrid = self.columnActive.reshape((self.height, self.width))

        return self.columnActive

if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 2
    minOverlap = 2

    # Some made up inputs to test with
    #colOverlapGrid = np.random.randint(10, size=(numRows, numCols))
    # colOverlapGrid = np.array([[8, 4, 5, 8],
    #                            [8, 6, 1, 6],
    #                            [7, 7, 9, 4],
    #                            [2, 3, 1, 5]])
    colOverlapGrid = np.array([[0, 5, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 6, 0]])
    potColOverlapGrid = np.array([[0, 5, 1, 0],
                                  [2, 1, 1, 0],
                                  [3, 0, 2, 3],
                                  [2, 0, 6, 0]])
    print "colOverlapGrid = \n%s" % colOverlapGrid
    print "potColOverlapGrid = \n%s" % potColOverlapGrid
    print "colActNotBurstGrid = \n%s" % colActNotBurstGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity,
                                           minOverlap,
                                           centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid, potColOverlapGrid)
    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns

    potColOverlapGrid = np.array([[0, 5, 1, 0],
                                  [3, 1, 1, 0],
                                  [3, 0, 2, 2],
                                  [2, 0, 6, 0]])

    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid, potColOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns
