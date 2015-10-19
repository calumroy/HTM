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

    def kthScore(self, overlapsVect, cols, kth):
        # print "overlapsVect = \n%s" % overlapsVect
        # print "cols = \n%s" % cols
        # print "kth = %s" % kth
        if len(cols) > 0 and kth > 0 and kth < (len(cols)-1):
            #Add the overlap values to a single list
            orderedScore = np.array([0 for i in range(len(cols))])
            for i in range(len(orderedScore)):
                orderedScore[i] = overlapsVect[cols[i]]
            #print cols[0].overlap
            orderedScore = np.sort(orderedScore)
            # print "orderedScore = \n%s" % orderedScore
            # print "orderedScore[-kth] = %s" % orderedScore[-kth]
            return orderedScore[-kth]       # Minus since list starts at lowest
        return 0

    def neighbours(self, overlapsGrid, pos_x, pos_y):
        # returns a list of the elements that are within the inhibitionRadius
        # of the element at the given position. The returned list is a
        # list of the indicies of the elements wen the overlaps grid ios flattened.
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
            topPos_y = self.potentialHeight/2
            bottomPos_y = int(math.ceil(self.potentialHeight/2.0))-1
            leftPos_x = self.potentialWidth/2
            rightPos_x = int(math.ceil(self.potentialWidth/2.0))-1

        # Add one to the c.pos_y+c.inhibitionRadius because for example range(0,2)=(0,1)
        for i in range(int(pos_y-topPos_y), int(pos_y+bottomPos_y)+1):
            if i >= 0 and i < self.height:
                for j in range(int(pos_x-leftPos_x), int(pos_x+rightPos_x)+1):
                    if j >= 0 and j < self.width:
                        closeColumns.append(i * self.width + j)
        return np.array(closeColumns)

    def calculateWinningCols(self, overlapsGrid):

        activeColumns = []
        assert self.width == len(overlapsGrid[0])
        assert self.height == len(overlapsGrid)
        allColsOverlaps = overlapsGrid.flatten().tolist()
        columnActive = np.zeros_like(allColsOverlaps)
        # print "columnActive = \n%s" % columnActive
        # print "overlapsGrid = \n%s" % overlapsGrid
        # print "allColsOverlaps = \n%s" % allColsOverlaps
        # Get all the columns in a 1D array then sort them based on their overlap value.
        # Return an array of the indicies containing the min overlap values to the max.
        sortedAllColsInd = np.argsort(allColsOverlaps)
        #sortedAllColsInd = np.flipud(allColsOverlaps)
        # print "sorted allColsOverlaps indicies = \n%s" % sortedAllColsInd

        # Now start from the columns with the highest overlap and inhibit
        # columns with smaller overlaps.
        for i in reversed(sortedAllColsInd):
            # Make sure the overlap value is larger then zero.
            overlap = allColsOverlaps[i]
            if overlap > 0:
                # Get the columns position
                pos_x = i % self.width
                pos_y = math.floor(i/self.height)

                #print "COLUMN INDEX = %s" % i
                #print "overlap = %s" % overlap
                # Get the neighbours of the column return the
                # indicies of the neighbouring columns
                neighbourCols = self.neighbours(overlapsGrid, pos_x, pos_y)
                minLocalActivity = self.kthScore(allColsOverlaps, neighbourCols, self.desiredLocalActivity)

                # print "neighbourCols = \n%s" % neighbourCols
                #print "minLocalActivity = %s" % minLocalActivity

                #import ipdb; ipdb.set_trace()
                if overlap > minLocalActivity:
                     # Activate the column and add the current time to the times array.
                    activeColumns.append(i)
                    columnActive[i] = 1
                elif overlap == minLocalActivity:
                    # Check the neighbours and see how many have an overlap
                    # larger then the minLocalctivity or are already active.
                    # These columns will be set active.
                    numActiveNeighbours = 0
                    for d in neighbourCols:
                        if (allColsOverlaps[d] > minLocalActivity or columnActive[d] == 1):
                            numActiveNeighbours += 1
                    # if less then the desired local activity have been set
                    # or will be set as active then activate this column as well.
                    if numActiveNeighbours < self.desiredLocalActivity:
                        activeColumns.append(i)
                        columnActive[i] = 1
                    else:
                        # Set the overlap score for the losing columns to zero
                        allColsOverlaps[i] = 0
                else:
                    # Set the overlap score for the losing columns to zero
                    allColsOverlaps[i] = 0
                #print "allColsOverlaps reshaped = \n%s" % np.array(allColsOverlaps).reshape((self.height, self.width))
                #print "columnActive reshaped = \n%s" % np.array(columnActive).reshape((self.height, self.width))

        # print "ACTIVE COLUMN INDICIES = \n%s" % activeColumns
        # print "columnActive = \n%s" % columnActive

        return columnActive

if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 5
    desiredLocalActivity = 2

    # Some made up inputs to test with
    #colOverlapGrid = np.random.randint(10, size=(numRows, numCols))
    colOverlapGrid = np.array([[8, 4, 5, 8, 1],
                               [8, 6, 1, 6, 1],
                               [7, 7, 9, 4, 9],
                               [2, 3, 1, 5, 9]])
    print "colOverlapGrid = \n%s" % colOverlapGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity, centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    activeColumns = activeColumns.reshape((numRows, numCols))
    print "activeColumns = \n%s" % activeColumns
