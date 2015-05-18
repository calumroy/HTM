# Title: HTM
# Description: git managed development of a HTM network
# Author: Calum Meiklejohn
# Development phase: alpha

import cProfile
import numpy as np
import random
import math
#import pprint
import copy
from utilities import sdrFunctions as SDRFunct
import Thalamus

##Struct = {'field1': 'some val', 'field2': 'some val'}
##myStruct = { 'num': 1}

SegmentUpdate = {'index': '-1',
                 'activeSynapses': '0',
                 'sequenceSegment': 0,
                 'createdAtTime': 0}


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


class Synapse:
    def __init__(self, input, pos_x, pos_y, cellIndex, permanence):
            # Cell is -1 if the synapse connects the HTM layers input.
            # Otherwise it is a horizontal connection to the cell.
            # The start is at a column or cells position the end is at
            # the coordinates stored in pos_x, pos_y.
            # Index self.cell in the column at self.pos_x self.pos_y
            self.cell = cellIndex
            # The end of the synapse is at this pos.
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.permanence = permanence


class Segment:
    def __init__(self):
        self.predict = False
        self.index = -1
        self.sequenceSegment = 0    # Stores the last time step that this segment was predicting activity
        # Stores the synapses that have been created and
        #have a larger permenence than 0.0
        self.synapses = []


class Cell:
    def __init__(self):
        # dendrite segments
        #self.numInitSegments = 1    # Must be greater then zero
        self.score = 0     # The current score for the cell.
        self.segments = []
        #for i in range(self.numInitSegments):
        #    self.segments = np.hstack((self.segments,[Segment()]))
        # Create a dictionary to store the segmentUpdate structures
        self.segmentUpdateList = []
        # Segment update stucture holds the updates for the cell.
        #These updates are made later.
        self.segmentUpdate = {'index': -1,
                              'activeSynapses': [],
                              'newSynapses': [],
                              'sequenceSegment': 0,
                              'createdAtTime': 0}
        #for i in range(self.numInitSegments):
        #    self.segmentUpdateList.append(self.segmentUpdate.copy())
        #print self.segmentUpdateList
##        # State of the cell
##        self.active = False
##        self.predict = False


class Column:
    def __init__(self, length, pos_x, pos_y, params):
        self.cells = [Cell() for i in range(length)]
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.overlap = 0.0  # As defined by the numenta white paper
        self.minOverlap = params['minOverlap']
        self.boost = params['boost']
        # The max distance a column can inhibit another column.
        self.inhibitionRadius = params['inhibitionRadius']
        # The max distance that Synapses can be made at
        self.potentialWidth = params['potentialWidth']
        self.potentialHeight = params['potentialHeight']
        # Spatial Pooler synapses inc or dec values
        self.spatialPermanenceInc = params['spatialPermanenceInc']
        self.spatialPermanenceDec = params['spatialPermanenceDec']
        # Sequence Pooling synapses inc or dec values
        self.permanenceInc = params['permanenceInc']
        self.permanenceDec = params['permanenceDec']
        self.minDutyCycle = params['minDutyCycle']   # The minimum firing rate of the column
        # Keeps track of when the column was active.
        # All columns start as active. It stores the
        #numInhibition time when the column was active
        self.activeDutyCycleArray = np.array([])
        self.activeDutyCycle = 0.0  # the firing rate of the column
        # The rate at which the overlap is larger then the min overlap
        self.overlapDutyCycle = 0.0
        # Keeps track of when the colums overlap was larger then the minoverlap
        self.overlapDutyCycleArray = np.array([0])
        # How much to increase the boost by when boosting is needed.
        self.boostStep = params['boostStep']
        # This determines how many previous timeSteps are stored in
        #actve predictive and learn state arrays.
        self.historyLength = params['historyLength']
        self.highestScoredCell = None
        # A time flag to indicate the column should stop temporally pooling
        # This is set to the current time when a column has a poor overlap value
        # but it is temporally pooling. On the next time step if the overlap is
        # still poor the column should not keep temporally pooling (being activated).
        self.stopTempAfterTime = -1
        # The last time temporal pooling occurred
        self.lastTempPoolingTime = -1

        # An array storing the synapses with a permanence greater then the connectPermanence.
        self.connectedSynapses = np.array([], dtype=object)

        # The possible feed forward Synapse connections for the column
        self.potentialSynapses = np.array([], dtype=object)

        # An array storing when each of the cells in the column were last in a predictive state.
        self.predictiveStateArray = np.array([0 for i in range(self.historyLength)])
        for i in range(length-1):   # Minus one since the first entry is already there
            self.predictiveStateArray = np.vstack((self.predictiveStateArray, [0 for i in range(self.historyLength)]))
        # An array storing the timestep when each cell in the column was last in
        # an active state. This means the column has
        # feedforward input and the cell has a temporal context indicated by
        # active segments.
        self.activeStateArray = np.array([0 for i in range(self.historyLength)])
        for i in range(length-1):
            self.activeStateArray = np.vstack((self.activeStateArray, [0 for i in range(self.historyLength)]))
        # An array storing when each of the cells in the column were last in a learn state.
        self.learnStateArray = np.array([0 for i in range(self.historyLength)])
        for i in range(length-1):
            self.learnStateArray = np.vstack((self.learnStateArray, [0 for i in range(self.historyLength)]))
        # An array storing the last scores for each of the cells
          # in the column.
        # The score is based on how many of the previous sequence patterns have been seen for a cell.
        #self.scoreArray = np.array([0 for i in range(self.historyLength)])
        #for i in range(length-1):   # Minus one since the first entry is already there
        #    self.scoreArray = np.vstack((self.scoreArray,[0 for i in range(self.historyLength)]))
        # An array storing the last timeSteps when the column was active.
        self.columnActive = np.array([0 for i in range(self.historyLength)])

    def updateBoost(self):
        if self.activeDutyCycle < self.minDutyCycle:
            self.boost = self.boost+self.boostStep
        else:
            #print "activeDutyCycle %s > minDutyCycle %s"
            # %(self.activeDutyCycle,self.minDutyCycle)
            self.boost = 1.0
        #print self.boost


class HTMLayer:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn, params):
        # The columns are in a 2 dimensional array columnArrayWidth by
        #columnArrayHeight.
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.Input = input
        # The overlap values are used in determining the active columns.
        # For columns with the same overlap value
        # both columns are active. This is why sometimes more columns then
        # the desiredLocalActivity parameter
        # are observed in the inhibition radius.
        # How many cells within the inhibition radius are active
        self.desiredLocalActivity = params['desiredLocalActivity']
        # If true then all the potential synapses for a column are centered
        # around the columns position else they are to the right of the columns pos.
        self.centerPotSynapses = params['centerPotSynapses']
        self.cellsPerColumn = cellsPerColumn
        # If the permanence value for a synapse is greater than this
        # value, it is said to be connected.
        self.connectPermanence = params['connectPermanence']
        # Should be smaller than activationThreshold
        self.minThreshold = params['minThreshold']
        # The minimum score needed by a cell to be added
        # to the alternative sequence.
        self.minScoreThreshold = params['minScoreThreshold']
        # This limits the activeSynapse array to this length. Should be renamed
        self.newSynapseCount = params['newSynapseCount']
        # The maximum number of segments allowed by a cell
        self.maxNumSegments = params['maxNumSegments']
        # More than this many synapses on a segment must be active for
        # the segment to be active
        self.activationThreshold = params['activationThreshold']
        self.dutyCycleAverageLength = params['dutyCycleAverageLength']
        self.timeStep = 0
        # The output is a 2D grid representing the cells states.
        # It is larger then the input by a factor of the number of cells per column
        self.output = np.array([[0 for i in range(self.width * self.cellsPerColumn)] for j in range(self.height)])
        self.activeColumns = np.array([], dtype=object)

        # The starting permance of new synapses. This is used to create new synapses.
        self.synPermanence = params['synPermanence']

        # Create the array storing the columns
        self.columns = np.array([[]], dtype=object)
        # Setup the columns array.
        self.setupColumns(params['Columns'])

    def setupColumns(self, columnParams):
        # Get just the parameters for the columns
        # Note: The parameters can come in a list of dictionaries,
        # one for each column or a shorter list specifying only some columns.
        # If only one or a few columns have parameters specified then all the
        # rest of the columns get the same last parameters specified.
        numColParams = len(columnParams)
        self.columns = np.array([[Column(self.cellsPerColumn,
                                         i,
                                         j,
                                         columnParams[min(i*self.width+j, numColParams-1)]
                                         )
                                for i in range(self.width)] for
                                j in range(self.height)], dtype=object)

        # Initialise the columns potential synapses.
        # Work out the potential feedforward connections each column could make to the input.
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                self.updatePotentialSynapses(c)

    def columnActiveNotBursting(self, col, timeStep):
        # Calculate which cell in a given column at the given time was active but not bursting.
        cellsActive = 0
        cellNumber = None
        for k in range(len(col.cells)):
            # Count the number of cells in the column that where active.
            if self.activeState(col, k, self.timeStep-1) is True:
                cellsActive += 1
                cellNumber = k
            if cellsActive > 1:
                break
        if cellsActive == 1 and cellNumber is not None:
            return cellNumber
        else:
            return None

    def activeCellGrid(self):
        # Return a grid representing the cells in the columns which are active.
        # Cells in a column are placed in adjacent grid cells right of each other.
        # Eg. A HTM layer with 10 rows, 5 columns and 3 cells per column would produce an
        # activeCellGrid of 10*3 = 15 columns and 10 rows.

        output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                          for j in range(self.height)])
        for c in self.activeColumns:
            x = c.pos_x
            y = c.pos_y
            for k in range(len(c.cells)):
                if c.activeStateArray[k][0] == self.timeStep:
                    output[y][x*self.cellsPerColumn+k] = 1
        return output

    def activeNotBurstCellGrid(self):
        # Return a grid representing the cells in the columns which are active but
        # not bursting. Cells in a column are placed in adjacent grid cells right of each other.
        # Eg. A HTM layer with 10 rows, 5 columns and 3 cells per column would produce an
        # activeCellGrid of 10*3 = 15 columns and 10 rows.
        output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                          for j in range(self.height)])
        for c in self.activeColumns:
            x = c.pos_x
            y = c.pos_y
            cellsActive = 0
            cellNumber = None
            for k in range(len(c.cells)):
                # Count the number of cells in the column that where active.
                if c.activeStateArray[k][0] == self.timeStep:
                    cellsActive += 1
                    cellNumber = k
                if cellsActive > 1:
                    break
            if cellsActive == 1 and cellNumber is not None:
                output[y][x*self.cellsPerColumn+cellNumber] = 1
        #print "output = ", output
        return output

    def predictiveCellGrid(self):
        # Return a grid representing the cells in the columns which are predicting.
        output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                          for j in range(self.height)])
        for y in range(len(self.columns)):
            for x in range(len(self.columns[i])):
                c = self.columns[y][x]
                for k in range(len(c.cells)):
                    # Count the number of cells in the column that are predicting now
                    if c.predictiveStateArray[k][0] == self.timeStep:
                        output[y][x*self.cellsPerColumn+k] = 1
        #print "output = ", output
        return output

    def updateOutput(self):
        # Update the output array.
        # The output array is the output from all the cells. The cells form a new 2d input grid
        # this way temporal information is not lost between layers and levels.
        # Initialise all outputs as zero first then set the cells as 1.
        for i in range(len(self.output)):
            for j in range(len(self.output[i])):
                self.output[i][j] = 0
        for c in self.activeColumns:
            x = c.pos_x
            y = c.pos_y
            # If the column is active now
            if self.columnActiveState(c, self.timeStep) is True:
                for i in range(self.cellsPerColumn):
                    # The first element is the last time the cells was active.
                    # If it equals the current time then the cell is active now.
                    if c.activeStateArray[i][0] == self.timeStep:
                        self.output[y][x*self.cellsPerColumn+i] = 1

    def updateConnectedSynapses(self, c):
        # Update the connectedSynapses array.
        c.connectedSynapses = np.array([], dtype=object)
        connSyn = []
        for i in range(len(c.potentialSynapses)):
            if c.potentialSynapses[i].permanence > self.connectPermanence:
                connSyn.append(c.potentialSynapses[i])
        c.connectedSynapses = np.append(c.connectedSynapses, connSyn)

    def changeColsPotRadius(self, newPotentialWidth, newPotentialHeight):
        # Change the potential radius of all the columns
        # This means the potential synapse list for all the
        # columns needs to be updated as well.
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                c.potentialWidth = newPotentialWidth
                c.potentialHeight = newPotentialHeight
                # Update the potential synapses since the potential radius has changed
                self.updatePotentialSynapses(c)

    def changeColsInhibRadius(self, newInhibRadius):
        # Change the inhibition radius of all the columns
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                c.inhibitionRadius = newInhibRadius

    def changeColsMinOverlap(self, newMinOverlap):
        # Change the columns minOverlap value for all columns.
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                c.minOverlap = newMinOverlap

    def changeColsSynSpatialDecrement(self, newPermanenceDec):
         # Change the columns spatial pooler permanence decrement value for all columns.
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                c.spatialPermanenceDec = newPermanenceDec

    def updatePotentialSynapses(self, c):
        # Update the locations of the potential synapses for column c.
        # If the input is larger than the number of columns then
        # the columns are evenly spaced out over the input.
        # First initialize the list to null
        c.potentialSynapses = np.array([])
        inputHeight = len(self.Input)
        inputWidth = len(self.Input[0])
        columnHeight = len(self.columns)
        columnWidth = len(self.columns[0])
        # Calculate the ratio between columns and the input space.
        colInputRatioHeight = float(inputHeight) / float(columnHeight)
        colInputRatioWidth = float(inputWidth) / float(columnWidth)
        # Cast the y position to an int so it matches up with a row number.
        inputCenter_y = int(c.pos_y*colInputRatioHeight)
        # Cast the x position to an int so it matches up with a column number.
        inputCenter_x = int(c.pos_x*colInputRatioWidth)

        # Define a range of pos values around the column depending on parameters.
        # This forms a rectangle of input squares that are covered by the pot synapses.
        if self.centerPotSynapses == 0:
            # This setting is used for the command space.
            topPos_y = inputCenter_y
            bottomPos_y = inputCenter_y+c.potentialHeight
            leftPos_x = inputCenter_x
            rightPos_x = inputCenter_x+c.potentialWidth
        else:
            # This setting is used for normal input space.
            topPos_y = inputCenter_y - c.potentialHeight/2
            bottomPos_y = inputCenter_y + c.potentialHeight/2 + 1
            leftPos_x = inputCenter_x - c.potentialWidth/2
            rightPos_x = inputCenter_x + c.potentialWidth/2 + 1

        for y in range(int(topPos_y),
                       int(bottomPos_y)):
            if y >= 0 and y < inputHeight:
                for x in range(int(leftPos_x),
                               int(rightPos_x)):
                    if x >= 0 and x < inputWidth:
                        # Create a Synapse pointing to the HTM layers input
                        #so the synapse cellIndex is -1
                        c.potentialSynapses = np.append(c.potentialSynapses,
                                                        [Synapse(self.Input, x, y, -1, self.synPermanence)])

    def neighbours(self, c):
        # returns a list of the columns that are within the inhibitionRadius of c
        closeColumns = []
        # Add one to the c.pos_y+c.inhibitionRadius because for example range(0,2)=(0,1)
        for i in range(int(c.pos_y-c.inhibitionRadius), int(c.pos_y+c.inhibitionRadius)+1):
            if i >= 0 and i < (len(self.columns)):
                for j in range(int(c.pos_x-c.inhibitionRadius), int(c.pos_x+c.inhibitionRadius)+1):
                    if j >= 0 and j < (len(self.columns[0])):
                        closeColumns.append(self.columns[i][j])
        return np.array(closeColumns)

    def areNeighbours(self, c, d):
        # Checks to see if two columns are neighbours.
        # This means the columns are within the inhibitionRadius of c.
        distance = int(math.sqrt(math.pow(d.pos_y - c.pos_y, 2) + math.pow(d.pos_x - c.pos_x, 2)))
        #print "c.inhibitionRadius = %s distance = %s" % (c.inhibitionRadius, distance)
        if distance <= c.inhibitionRadius:
            return True
        else:
            return False

    def updateOverlapDutyCycle(self, c):
            # Append the current time to the list of times that the column was active for
            c.overlapDutyCycleArray = np.append(c.overlapDutyCycleArray, self.timeStep)
            for i in range(len(c.overlapDutyCycleArray)):        # Remove the values that where too long ago
                if c.overlapDutyCycleArray[0] < (self.timeStep-self.dutyCycleAverageLength):
                    c.overlapDutyCycleArray = np.delete(c.overlapDutyCycleArray, 0, 0)
                else:
                    break
            #Update the overlap duty cycle running average
            c.overlapDutyCycle = float(len(c.overlapDutyCycleArray))/float(self.dutyCycleAverageLength)
            #print "overlap DutyCycle = %s length =
            # %s averagelength = %s"%(c.overlapDutyCycle,len
                # (c.overlapDutyCycleArray),self.dutyCycleAverageLength)

    def increasePermanence(self, c, scale):
        # Increases all the permanences of the Synapses.
        # It's used to help columns win that don't
        # have a good overlap with any inputs
        for i in range(len(c.potentialSynapses)):
            # Increase the permance by a scale factor
            c.potentialSynapses[i].permanence = min(1.0, (1+scale)*(c.potentialSynapses[i].permanence))

    def kthScore(self, cols, kth):
        if len(cols) > 0 and kth > 0 and kth < (len(cols)-1):
            #Add the overlap values to a single list
            orderedScore = np.array([0 for i in range(len(cols))])
            for i in range(len(orderedScore)):
                orderedScore[i] = cols[i].overlap
            #print cols[0].overlap
            orderedScore = np.sort(orderedScore)
            #print orderedScore
            return orderedScore[-kth]       # Minus since list starts at lowest
        return 0

    def updateActiveDutyCycle(self, c):
        # If the column is active now
        if self.columnActiveState(c, self.timeStep) is True:
            # Append the current time call to the list of times that the column was active for
            c.activeDutyCycleArray = np.append(c.activeDutyCycleArray, self.timeStep)
        for i in range(len(c.activeDutyCycleArray)):
        # Remove the values that where too long ago
            if c.activeDutyCycleArray[0] < (self.timeStep-self.dutyCycleAverageLength):
                c.activeDutyCycleArray = np.delete(c.activeDutyCycleArray, 0, 0)
            else:
                break
        #Update the active duty cycle running average
        c.activeDutyCycle = float(len(c.activeDutyCycleArray))/float(self.dutyCycleAverageLength)
        #print "DutyCycle = %s length = %s averagelength
        # = %s"%(c.activeDutyCycle,len(c.activeDutyCycleArray),
            # self.dutyCycleAverageLength)

    def maxDutyCycle(self, cols):
        maxActiveDutyCycle = 0.0
        for c in cols:
            if maxActiveDutyCycle < c.activeDutyCycle:
                maxActiveDutyCycle = c.activeDutyCycle
        return maxActiveDutyCycle

    def deleteSegments(self, c, i):
        # Delete the segments that have no synapses in them.
        # Also delete the most unused segments if too many exist.
        # This should only be done before or after learning since a
        # segments index in a cells segments list is used for learning.
        deleteSegments = []
        # This is a list of the indicies of the segments that will be deleted
        #print "Delete seg in Cell i = %s" % i
        for s in range(len(c.cells[i].segments)):
            if len(c.cells[i].segments[s].synapses) == 0:
                deleteSegments.append(c.cells[i].segments[s])
        for d in deleteSegments:
            c.cells[i].segments.remove(d)

        deleteSegments = []
        numSegments = len(c.cells[i].segments)
        if numSegments > self.maxNumSegments:
            # The max number of segments allowed has been reached.
            # Find the most unused segment and delete it.
            deleteSegments = self.findLeastUsedSeg(c, i)
        for d in deleteSegments:
            c.cells[i].segments.remove(d)

    def findLeastUsedSeg(self, c, i):
        # Find the most unused segment from the column c cell i
        # segments list. Return it in a list.
        leastUsedSeg = []
        unusedSegment = None
        oldestTime = None

        for s in c.cells[i].segments:
            if (((s.sequenceSegment < oldestTime) and s.sequenceSegment != 0)
               or oldestTime is None):
                oldestTime = s.sequenceSegment
                unusedSegment = s
        leastUsedSeg = [unusedSegment]
        return leastUsedSeg

    def deleteWeakSynapses(self, c, i, segIndex):
        # Delete the synapses that have a permanence
        # value that is too low from the segment.
        deleteActiveSynapses = []
        # This is a list of the indicies of the
        # active synapses that will be deleted
        for syn in c.cells[i].segments[segIndex].synapses:
            if syn.permanence < self.connectPermanence:
                deleteActiveSynapses.append(syn)
        for d in deleteActiveSynapses:
            c.cells[i].segments[segIndex].synapses.remove(d)
        #print "     deleted %s number of synapses"%(len(deleteActiveSynapses))

    def columnActiveAdd(self, c, timeStep):
        # We add the new time to the start of the array
        # then delete the time at the end of the array.
        # All the times should be in order from
        # most recent to oldest.
        newArray = np.insert(c.columnActive, 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.columnActive = newArray

    def activeStateAdd(self, c, i, timeStep):
        # We add the new time to the start of
        # the array then delete the time at the end of the array.
        # All the times should be in order from
        # most recent to oldest.
        newArray = np.insert(c.activeStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.activeStateArray[i] = newArray

    def predictiveStateAdd(self, c, i, timeStep):
        # We add the new time to the start of the array
        # then delete the time at the end of the array.
        # All the times should be in order from
        # most recent to oldest.
        newArray = np.insert(c.predictiveStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.predictiveStateArray[i] = newArray

    def learnStateAdd(self, c, i, timeStep):
        # We add the new time to the start of the
        # array then delete the time at the end of the array.
        # All the times should be in order from
        # most recent to oldest.
        newArray = np.insert(c.learnStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.learnStateArray[i] = newArray

    def columnActiveState(self, c, timeStep):
        # Search the history of the columnActive to find if the
        # column was active at time timeStep
        for j in range(len(c.columnActive)):
            if c.columnActive[j] == timeStep:
                return True
        return False

    def activeState(self, c, i, timeStep):
        # Search the history of the activeStateArray to find if the
        # cell was active at time timeStep
        for j in range(len(c.activeStateArray[i])):
            if c.activeStateArray[i, j] == timeStep:
                return True
        return False

    def predictiveState(self, c, i, timeStep):
        # Search the history of the predictiveStateArray to find if the
        # cell was predicting at time timeStep
        for j in range(len(c.predictiveStateArray[i])):
            if c.predictiveStateArray[i, j] == timeStep:
                return True
        return False

    def learnState(self, c, i, timeStep):
        # Search the history of the learnStateArray to find if the
        # cell was learning at time timeStep
        for j in range(len(c.learnStateArray[i])):
            if c.learnStateArray[i, j] == timeStep:
                return True
        return False

    def findActiveCell(self, c, timeStep):
        # Return the cell index that was active in the column c at
        # the timeStep provided.
        # If all cells are active (ie bursting was occuring) then return
        # all indicies. If no cell was active then return None
        activeCellsArray = []
        for i in range(self.cellsPerColumn):
            if self.activeState(c, i, timeStep) is True:
                activeCellsArray.append(i)
        return activeCellsArray

    def findLearnCell(self, c, timeStep):
        # Return the cell index that was in the learn state in the column c at
        # the timeStep provided.
        # If no cell was in the learn state then return None
        learnCell = None
        for i in range(self.cellsPerColumn):
            if self.learnState(c, i, timeStep) is True:
                learnCell = i
        return learnCell

    def randomActiveSynapses(self, c, i, s, timeStep):
        # Randomly add self.newSynapseCount-len(synapses) number of Synapses
        # that connect with cells that are active
        #print "randomActiveSynapses time = %s"%timeStep
        synapseList = []
        # A list of all potential synapses that are active
        for l in range(len(self.columns)):
        # Can't use c since c already represents a column
            for m in self.columns[l]:
                for j in range(len(m.learnStateArray)):
                    if self.learnState(m, j, timeStep) is True:
                        #print "time = %s synapse ends at
                        # active cell x,y,i = %s,%s,%s"%(timeStep,m.pos_x,m.pos_y,j)
                        synapseList.append(Synapse(0, m.pos_x, m.pos_y, j, self.synPermanence))
        # Take a random sample from the list synapseList
        # Check that there is at least one segment
        # and the segment index isnot -1 meaning
        # it's a new segment that hasn't been created yet.
        if len(c.cells[i].segments) > 0 and s != -1:
            numNewSynapses = self.newSynapseCount-len(c.cells[i].segments[s].synapses)
        else:
            numNewSynapses = self.newSynapseCount
        # Make sure that the number of new synapses
        # to choose isn't larger than the
        #total amount of active synapses to choose from but is larger than zero.
        if numNewSynapses > len(synapseList):
            numNewSynapses = len(synapseList)
        if numNewSynapses <= 0:
            numNewSynapses = 0
            #print "%s new synapses from len(synList) =
            # %s" %(numNewSynapses,len(synapseList))
            # return an empty list. This means this
            # segment has too many synapses already
            return []
        return random.sample(synapseList, numNewSynapses)

    def getActiveSegment(self, c, i, t):
        # Returns a sequence segment if there are none
        # then returns the most active segment
        highestActivity = 0
        mostActiveSegment = -1
        for s in c.cells[i].segments:
            activeState = 1
            activity = self.segmentActive(s, t, activeState)
            if s.sequenceSegment == t:
                #print "RETURNED SEQUENCE SEGMENT"
                return s
            else:
                mostActiveSegment = s
                if activity > highestActivity:
                    highestActivity = activity
                    mostActiveSegment = s
        return mostActiveSegment

    def segmentHighestScore(self, s, timeStep):
        # Only cells scores that are in active columns
        # in the current timeStep are checked.
        # Cells score are updated whenever they are in an
        # active column. This prevents scores getting stale.
        highestScoreCount = 0
        for i in range(len(s.synapses)):
            x = s.synapses[i].pos_x
            y = s.synapses[i].pos_y
            cell = s.synapses[i].cell
            if self.columnActiveState(self.columns[y][x], timeStep) is True:
                if self.columns[y][x].cells[cell].score > highestScoreCount:
                    highestScoreCount = self.columns[y][x].cells[cell].score
        return highestScoreCount

    def segmentActive(self, s, timeStep, state):
        # For Segment s check if the number of
        # synapses with the state "state" is larger then
        # the self.activationThreshold.
        # state is -1 = predictive state, 1 = active, 2 = learn state
        count = 0
        for i in range(len(s.synapses)):
            # Only check synapses that have a large enough permanence
            if s.synapses[i].permanence > self.connectPermanence:
                x = s.synapses[i].pos_x
                y = s.synapses[i].pos_y
                cell = s.synapses[i].cell
                if state == 1:  # 1 is active state
                    if self.activeState(self.columns[y][x], cell, timeStep) is True:
                        count += 1
                elif state == -1:  # -1 is predictive state
                    if self.predictiveState(self.columns[y][x], cell, timeStep) is True:
                        count += 1
                elif state == 2:    # 2 is learn state
                    if self.learnState(self.columns[y][x], cell, timeStep) is True:
                        count += 1
                else:
                    print "ERROR state is not a -1 predictive or 1 active or 2 learn"
        #if state==1:    # Used for printing only
        #    print" count = %s activeThreshold=%s"%
        # (count, self.activationThreshold)
        if count > self.activationThreshold:
            #print"         %s synapses were active on segment"%count
            # If the state is active then those synapses
            # in the segment have activated the
            # segment as being a sequence segment i.e.
            # the segment is predicting that the cell
            # will be active on the next time step.
            if state == 1:  # 1 is active state
                s.sequenceSegment = timeStep
            return count
        else:
            return 0

    def segmentNumSynapsesActive(self, s, timeStep, onCell):
        # For Segment s find the number of active synapses.
        # Synapses whose end is on an active cell or column. If the onCell is
        # true then we find the synapses that end on active cells.
        # If the onCell is false we find the synapses
        # that end on a column that is active.
        count = 0
        for i in range(len(s.synapses)):
            x = s.synapses[i].pos_x
            y = s.synapses[i].pos_y
            cell = s.synapses[i].cell
            if onCell is True:
                if self.activeState(self.columns[y][x], cell, timeStep) is True:
                    count += 1
            else:
                if self.columnActiveState(self.columns[y][x], timeStep) is True:
                    count += 1
        return count

    def getBestMatchingSegment(self, c, i, timeStep, onCell):
        # This routine is agressive. The permanence value is allowed to be less
        # then connectedPermance and activationThreshold > number of active Synpses > minThreshold
        # We find the segment who was most predicting for the current timestep and call
        # this the best matching segment.
        # This means we need to find synapses that where active at timeStep.
        # Note that this function is already called with time timeStep-1
        h = 0   # mostActiveSegmentIndex
        # Look through the segments for the one with the most active synapses
        #print "getBestMatchingSegment for x,y,c =
        #%s,%s,%s num segs = %s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments))
        for g in range(len(c.cells[i].segments)):
            # Find synapses that are active at timeStep
            currentSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[g], timeStep, onCell)
            mostActiveSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, onCell)
            if currentSegSynCount > mostActiveSegSynCount:
                h = g
                #print "\n new best matching segment found for h = %s\n"%h
                #print "segIndex = %s num of syn = %s num active syn =
                #"%(h,len(c.cells[i].segments[h].synapses),currentSegSynCount)
                #print "segIndex = %s"%(h)
        # Make sure the cell has at least one segment
        if len(c.cells[i].segments) > 0:
            if self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, onCell) > self.minThreshold:
                #print "returned the segment index (%s) which
                #HAD MORE THAN THE MINTHRESHOLD SYNAPSES"%h
                return h    # returns just the index to the
                #most active segment in the cell
        #print "returned no segment. None had enough active synapses return -1"
        return -1   # -1 means no segment was active
        #enough and a new one will be created.

    def getBestMatchingCell(self, c, timeStep):
        # Return the cell and the segment that is most matching in the column.
        # If no cell has a matching segment (no segment has more
        # then minThreshold synapses active)
        # then return the cell with the fewest segments
        # Nupic doen't return the cell with the fewest segments.
        bestCellFound = False
        # A flag to indicate that a
        #bestCell was found. A cell with at least one segment.
        bestCell = 0   # Cell index with the most active Segment
        bestSegment = 0
        # The segment index for the most active segment
        fewestSegments = 0
        # The cell index of the cell with the
        #least munber of segments
        h = 0           # h is the SegmentIndex of the most
        #active segment for the current cell i
        #print "getBestMatchingCell for x,y = %s,%s"%(c.pos_x,c.pos_y)
        for i in range(self.cellsPerColumn):
            # Find the cell index with the fewest number of segments.
            if len(c.cells[i].segments) < len(c.cells[fewestSegments].segments):
                fewestSegments = i
            h = self.getBestMatchingSegment(c, i, timeStep, True)
            if h >= 0:
                # Need to make sure the best cell actually has a segment.
                if len(c.cells[bestCell].segments) > 0:
                    #print "Best Segment at the moment is segIndex=%s"%bestSegment
                    # Must be larger than or equal to otherwise
                    # cell 0 segment 0 will never be chosen as the best cell
                    if (self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, True) >= self.segmentNumSynapsesActive(c.cells[bestCell].segments[bestSegment], timeStep, True)):
                        # Remember the best cell and segment
                        bestCell = i
                        bestSegment = h
                        bestCellFound = True
        if bestCellFound is True:
            #print "returned from GETBESTMATCHINGCELL the cell i=%s with the best segment s=%s"%(bestCell,bestSegment)
            return (bestCell, bestSegment)
        else:
            # Return the first segment from the cell with the fewest segments
            #print "returned from getBestMatchingCell cell i=%s with the fewest number of segments num=%s"%(fewestSegments,len(c.cells[fewestSegments].segments))
            return (fewestSegments, -1)

    def getSegmentActiveSynapses(self, c, i, timeStep, s, newSynapses=False):
        # Returns an segmentUpdate structure. This is used
        #to update the segments and their
        # synapses during learning. It adds the synapses
        #from the segments synapse list
        # that have an active end, to the segmentUpdate
        # structure so these synapses can be updated
        # appropriately (either inc or dec) later during learning.
        # s is the index of the segment in the cells segment list.
        newSegmentUpdate = {'index': s, 'activeSynapses': [],
                            'newSynapses': [], 'sequenceSegment': 0,
                            'createdAtTime': timeStep}
        #print "    getSegmentActiveSynapse called for
        #timeStep = %s x,y,i,s = %s,%s,%s,%s newSyn =
        #%s"%(timeStep,c.pos_x,c.pos_y,i,s,newSynapses)
        # If segment exists then go through an see which synapses are active.
        #Add them to the update structure.
        if s != -1:
            if len(c.cells[i].segments) > 0:
            # Make sure the array isn't empty
                if len(c.cells[i].segments[s].synapses) > 0:
                    for k in range(len(c.cells[i].segments[s].synapses)):
                        end_x = c.cells[i].segments[s].synapses[k].pos_x
                        end_y = c.cells[i].segments[s].synapses[k].pos_y
                        end_cell = c.cells[i].segments[s].synapses[k].cell
                        #print "Synapse ends at (%s,%s,%s)"
                        #%(end_x, end_y, end_cell)
                        # Check to see if the Synapse is
                        #connected to an active cell
                        if self.activeState(self.columns[end_y][end_x], end_cell, timeStep) is True:
                            # If so add it to the updateSegment structure
                            #print "     active synapse starts at
                            #x,y,cell,segIndex = %s,%s,%s,%s"%
                            #(c.pos_x,c.pos_y,i,s)
                            #print "     the synapse ends at x,y,cell
                            #= %s,%s,%s"%(end_x,end_y,end_cell)
                            newSegmentUpdate['activeSynapses'].append(c.cells[i].segments[s].synapses[k])
        if newSynapses is True:
            # Add new synapses that have an active end.
            # We add them  to the segmentUpdate structure. They are added
            # To the actual segment later during learning when the cell is
            # in a learn state.
            newSegmentUpdate['newSynapses'].extend(self.randomActiveSynapses(c, i, s, timeStep))
            #print "     New synapse added to the segmentUpdate"
        # Return the update structure.
        #print "     returned from getSegmentActiveSynapse"
        return newSegmentUpdate

    def adaptSegments(self, c, i, positiveReinforcement):
        #print " adaptSegments x,y,cell = %s,%s,%s positive
        #reinforcement = %r"%(c.pos_x,c.pos_y,i,positiveReinforcement)
        # Adds the new segments to the cell and inc or dec the segments synapses
        # If positive reinforcement is true then segments on the update list
        # get their permanence values increased all others
        #get their permanence decreased.
        # If positive reinforcement is false then decrement
        #the permanence value for the active synapses.
        for j in range(len(c.cells[i].segmentUpdateList)):
            #print "     segUpdateList = %s" % c.cells[i].segmentUpdateList[j]
            segIndex = c.cells[i].segmentUpdateList[j]['index']
            #print "     segIndex = %s"%segIndex
            # If the segment exists
            if segIndex > -1:
                #print "     adapted x,y,cell,segment=%s,%s,%s,%s"%(c.pos_x,c.pos_y,i,c.cells[i].segmentUpdateList[j]['index'])
                for s in c.cells[i].segmentUpdateList[j]['activeSynapses']:
                    # For each synapse in the segments activeSynapse list increment or
                    # decrement their permanence values.
                    # The synapses in the update segment
                    #structure are already in the segment. The
                    # new synapses are not yet however.
                    if positiveReinforcement is True:
                        s.permanence += c.permanenceInc
                        s.permanence = min(1.0, s.permanence)
                    else:
                        s.permanence -= c.permanenceDec
                        s.permanence = max(0.0, s.permanence)
                    #print "     x,y,cell,segment= %s,%s,%s,%s
                    #syn end x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,
                        #c.cells[i].segmentUpdateList[j]['index'],s.pos_x,
                        #s.pos_y,s.cell)
                    #print "     synapse permanence = %s"%(s.permanence)
                # Decrement the permanence of all synapses in the synapse list
                for s in c.cells[i].segments[segIndex].synapses:
                    s.permanence -= c.permanenceDec
                    s.permanence = max(0.0, s.permanence)
                    #print "     x,y,cell,segment= %s,%s,%s,%s syn end x,
                    #y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,j,
                        #s.pos_x,s.pos_y,s.cell)
                    #print "     synapse permanence = %s"%(s.permanence)
                # Add the new Synpases in the structure to the real segment
                #print c.cells[i].segmentUpdateList[j]['newSynapses']
                #print "oldActiveSyn = %s newSyn = %s"
                #%(len(c.cells[i].segments[segIndex].synapses),
                    #len(c.cells[i].segmentUpdateList[j]['newSynapses']))
                c.cells[i].segments[segIndex].synapses.extend(c.cells[i].segmentUpdateList[j]['newSynapses'])
                # Delete synapses that have low permanence values.
                self.deleteWeakSynapses(c, i, segIndex)
            # If the segment is new (the segIndex = -1) add it to the cell
            else:
                newSegment = Segment()
                newSegment.synapses = c.cells[i].segmentUpdateList[j]['newSynapses']
                c.cells[i].segments.append(newSegment)
                #print "     new segment added for x,y,cell,seg =%s,%s,%s,%s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments)-1)
                #for s in c.cells[i].segments[-1].synapses:
                #    print "Synapse added = %s" % s
                # Used for printing only
                #    print "         synapse ends at x,y=%s,%s" % (s.pos_x, s.pos_y)
        c.cells[i].segmentUpdateList = []
        # Delete the list as the updates have been added.

    def updateInput(self, newInput):
        # Update the input
        #Check to see if this input is the same size as the last and is a 2d numpy array
        try:
            assert len(newInput.shape) == 2
            assert newInput.shape == self.Input.shape
            self.Input = newInput
        except ValueError:
            print "New Input is not a 2D numpy array!"

    def temporalPool(self, c):
        # Temporal pooling is done here by increasing the overlap if
        # potential synapses are connected to active inputs
        # Add the potenial (2*radius+1)^2 as this is the maximum
        # overlap a column could have. This guarantees the column will
        # win later when inhibition occurs.

        if c.overlap >= c.minOverlap:
            # The col has a good overlap value and should allow temp pooling
            # to continue on the next time step. Set the time flag to not the
            # current time to allow this (we'll use zero).
            c.stopTempAfterTime = 0

        # If the time flag for temporal pooling was not set one time step ago
        # the we should perform temporal pooling.
        if c.stopTempAfterTime != (self.timeStep - 1):
            if c.overlap < c.minOverlap:
                # The current col has a poor overlap and should stop temporal
                # pooling on the timestep.
                c.stopTempAfterTime = self.timeStep

        # If more potential synapses then the min overlap
        # are active then set the overlap to the maximum value possible.
        if c.overlap >= c.minOverlap:
            maxOverlap = (c.potentialWidth)*(c.potentialHeight)
            c.overlap = c.overlap + maxOverlap
            c.lastTempPoolingTime = self.timeStep

    def Overlap(self):
        """
        Phase one for the spatial pooler

        This function also includes the new temporal pooling agorithm.
        The temporal pooler works in the spatial pooler by keeping columns
        active that where active but not bursting on the previous time step.

        This is done so the columns learn to activate on multiple input patterns
        keeping them activated throughout a sequence of patterns.

        Note the temporal pooling is very dependent on the potential radius.
        a larger potential radius allows the temporal pooler to pool over a larger
        number of columns. Columns temporally pool using their potential synapses,
        where normal colum activation only counts connected synapses.
        """
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.overlap = 0.0
                self.updateConnectedSynapses(c)

                # Calculate the overlap
                for s in c.connectedSynapses:
                    # Check if the input that this synapses
                    # is connected to is active.
                    #print "s.pos_y = %s s.pos_x = %s" % (s.pos_y, s.pos_x)
                    #print "input width = %s input height = %s" % (len(self.Input[0]), len(self.Input))
                    inputActive = self.Input[s.pos_y][s.pos_x]
                    c.overlap = c.overlap + inputActive

                # Temporal pooling is done if the column was active but not bursting one timestep ago.
                if self.columnActiveNotBursting(c, self.timeStep-1) is not None:
                    self.temporalPool(c)

                if c.overlap < c.minOverlap:
                    c.overlap = 0.0
                else:
                    c.overlap = c.overlap*c.boost
                    self.updateOverlapDutyCycle(c)
                #print "%d %d %d" %(c.overlap,c.minOverlap,c.boost)

    def inhibition(self, timeStep):
        '''
        Phase two for the spatial pooler

        Inhibit the weaker active columns.
        '''
        #print "length active columns before deleting = %s" % len(self.activeColumns)
        self.activeColumns = np.array([], dtype=object)
        #print "actve cols before %s" %self.activeColumns

        for i in range(len(self.columns)):
            for c in self.columns[i]:
                if c.overlap > 0:
                    minLocalActivity = self.kthScore(self.neighbours(c), self.desiredLocalActivity)
                    #print "current column = (%s,%s)"%(c.pos_x,c.pos_y)
                    if c.overlap > minLocalActivity:
                        self.activeColumns = np.append(self.activeColumns, c)
                        self.columnActiveAdd(c, timeStep)
                        #print "ACTIVE COLUMN x,y = %s, %s overlap = %d min = %d" % (c.pos_x, c.pos_y,
                        #                                                            c.overlap, minLocalActivity)
                    if c.overlap == minLocalActivity:
                        # Check the neighbours and see how many have an overlap
                        # larger then the minLocalctivity or are already active.
                        # These columns will be set active.
                        numActiveNeighbours = 0
                        for d in self.neighbours(c):
                            if (d.overlap > minLocalActivity or self.columnActiveState(d, self.timeStep) is True):
                                numActiveNeighbours += 1
                        # if less then the desired local activity have been set
                        # or will be set as active then activate this column as well.
                        if numActiveNeighbours < self.desiredLocalActivity:
                            #print "Activated column x,y = %s, %s numActiveNeighbours = %s" % (c.pos_x, c.pos_y, numActiveNeighbours)
                            self.activeColumns = np.append(self.activeColumns, c)
                            self.columnActiveAdd(c, timeStep)
                self.updateActiveDutyCycle(c)
                # Update the active duty cycle variable of every column

    def learning(self):
        '''
        Phase three for the spatial pooler

        Update the column synapses permanence.
        '''
        for c in self.activeColumns:
            for s in c.potentialSynapses:
                # Check if the input that this
                #synapses is connected to is active.
                inputActive = self.Input[s.pos_y][s.pos_x]
                if inputActive == 1:
                #Only handles binary input sources
                    s.permanence += c.spatialPermanenceInc
                    s.permanence = min(1.0, s.permanence)
                else:
                    s.permanence -= c.spatialPermanenceDec
                    s.permanence = max(0.0, s.permanence)

        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.minDutyCycle = 0.01*self.maxDutyCycle(self.neighbours(c))
                c.updateBoost()

                if c.overlapDutyCycle < c.minDutyCycle:
                    self.increasePermanence(c, 0.1*self.connectPermanence)


    def updateActiveState(self, timeStep):
        # First function called to update the sequence pooler.
        """ This function has been modified to the CLA whitepaper but it resembles
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
                    #print "Length of cells updatelist =
                    #%s"%len(c.cells[cell].segmentUpdateList)

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
                    #the column as the predicting cell.
                    #print "time = %s segment x,y,cell,
                    #segindex = %s,%s,%s,%s is active and NOW
                    #PREDICTING"%(timeStep,c.pos_x,c.pos_y,mostPredCell,
                        #mostPredSegment)
                    self.predictiveStateAdd(c, mostPredCell, timeStep)
                    activeUpdate = self.getSegmentActiveSynapses(c, mostPredCell, timeStep, mostPredSegment, False)
                    c.cells[mostPredCell].segmentUpdateList.append(activeUpdate)
                    # This differs to the CLA. All our segments are only active
                    # when in a predicting state so we don't need predSegment.
                    #predSegment=self.getBestMatchingSegment(c,i,timeStep-1)
                    #predUpdate=self.getSegmentActiveSynapses
                    #(c,i,timeStep-1,predSegment,True)
                    #c.cells[i].segmentUpdateList.append(predUpdate)

    def sequenceLearning(self, timeStep):
        # Third function called for the sequence pooler.
        # The update structures are implemented on the cells
        #print "\n       3rd SEQUENCE FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    #print "predictiveStateArray for x,y,i =
                    #%s,%s,%s is latest time = %s"%(c.pos_x,c.pos_y,i,
                        #c.predictiveStateArray[i,0])
                    if ((self.learnState(c, i, timeStep) is True)
                            and (self.learnState(c, i, timeStep-1) is False)):
                        #print "learn state for x,y,cell =
                        #%s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, True)
                    # Trying a different method to the CLA white pages
                    #if self.activeState(c,i,timeStep) ==
                    #False and self.predictiveState(c,i,timeStep-1) is True:
                    if ((self.predictiveState(c, i, timeStep-1) is True
                            and self.predictiveState(c, i, timeStep) is False
                            and self.activeState(c, i, timeStep) is False)):
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


class HTMRegion:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn, params):
        '''
        The HTMRegion is an object holding multiple HTMLayers. The region consists of
        simulated cortex layers.

        The lowest layer recieves the new input and feedback from the higher levels.

        The highest layer is a command/input layer. It recieves input from the
        lowest layers and from an outside thalamus class.
        This extra input is meant to direct the HTM.
        '''
        # The class contains multiple HTM layers stacked on one another
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.cellsPerColumn = cellsPerColumn
        self.numLayers = params['numLayers']  # The number of HTM layers that make up a region.
        self.layerArray = np.array([], dtype=object)
        # Make a place to store the thalamus command.
        self.commandInput = np.array([[0 for i in range(self.width*cellsPerColumn)]
                                     for j in range(self.height)])
        # Setup space in the input for a command feedback SDR
        self.enableCommandFeedback = params['enableCommandFeedback']

        self.setupLayers(input, params['HTMLayers'])

        # create and store a thalamus class if the
        # command feedback param is true.
        self.thalamus = None
        if self.enableCommandFeedback == 1:
            # The width of the the thalamus should match the width of the input grid.
            thalamusParams = params['Thalamus']
            self.thalamus = Thalamus.Thalamus(self.width*self.cellsPerColumn,
                                              self.height,
                                              thalamusParams)

    def setupLayers(self, input, htmLayerParams):
        # Set up the inputs to the HTM layers.
        # Note the params comes in a list of dics, one for each layer.
        # Layer 0 gets the new input.
        bottomLayerParams = htmLayerParams[0]
        self.layerArray = np.append(self.layerArray, HTMLayer(input,
                                                              self.width,
                                                              self.height,
                                                              self.cellsPerColumn,
                                                              bottomLayerParams))
        # The higher layers receive the lower layers output.
        for i in range(1, self.numLayers):
            # Try to get the parameters for this layer else use the
            # last specified params from the highest layer.
            if len(htmLayerParams) >= i+1:
                layersParams = htmLayerParams[i]
            else:
                layersParams = htmLayerParams[-1]

            lowerOutput = self.layerArray[i-1].output

            # The highest layer receives the lower layers input and
            # an input from the thalamus equal in size to the lower layers input.
            highestLayer = self.numLayers - 1
            if i == highestLayer and self.enableCommandFeedback == 1:
                lowerOutput = SDRFunct.joinInputArrays(self.commandInput, lowerOutput)
            self.layerArray = np.append(self.layerArray,
                                        HTMLayer(lowerOutput,
                                                 self.width,
                                                 self.height,
                                                 self.cellsPerColumn,
                                                 layersParams))

    def updateThalamus(self):
        # Update the thalamus.
        # This updates the command input that comes from
        # the thalamus.
        # Get the predicted command from the command space.
        # Pass this to the thalamus
        if self.thalamus is not None:
            topLayer = self.numLayers-1
            predCommGrid = self.layerPredCommandOutput(topLayer)
            #print "predCommGrid = %s" % predCommGrid
            thalamusCommand = self.thalamus.pickCommand(predCommGrid)

            # Update each level of the htm with the thalamus command
            self.updateCommandInput(thalamusCommand)

    def rewardThalamus(self, reward):
        # Reward the Thalamus.
        if self.thalamus is not None:
            self.thalamus.rewardThalamus(reward)

    def updateCommandInput(self, newCommand):
        # Update the command input for the level
        # This input is used by the top layer in this level.
        self.commandInput = newCommand

    def updateRegionInput(self, input):
        # Update the input and outputs of the layers.
        # Layer 0 receives the new input.
        highestLayer = self.numLayers - 1

        self.layerArray[0].updateInput(input)

        # The middle layers receive inputs from the lower layer outputs
        for i in range(1, self.numLayers):
            lowerOutput = self.layerArray[i - 1].output
            # The highest layer receives the lower layers input and
            # the commandInput for the level, equal in size to the lower layers input.
            if i == highestLayer and self.enableCommandFeedback == 1:
                lowerOutput = SDRFunct.joinInputArrays(self.commandInput, lowerOutput)
            self.layerArray[i].updateInput(lowerOutput)

    def layerOutput(self, layer):
        # Return the output for the given layer.
        return self.layerArray[layer].output

    def regionOutput(self):
        # Return the output form the entire region.
        # This will be the output from the highest layer.
        highestLayer = self.numLayers - 1
        return self.layerOutput(highestLayer)

    def layerPredCommandOutput(self, layer):
        # Return the given layers predicted command output.
        # This is the predictive cells from the command space.
        # The command space is assumed to be the top half of the
        # columns.

        # Divide the predictive cell grid into two and take the
        # top most part which is the command space.
        totalPredGrid = self.layerArray[layer].predictiveCellGrid()
        # Splits the grid into 2 parts of equal or almost equal size.
        # This splits the top and bottom. Return the top at index[0].
        PredGridOut = np.array_split(totalPredGrid, 2)[0]

        return PredGridOut

    def layerActCommandOutput(self, layer):
        # Return the given layers active command output.
        totalActiveGrid = self.layerArray[layer].activeCellGrid()
        # This is the active cells from the command space.
        # The command space is assumed to be the top half of the
        # columns.
        # Divide the active cell grid into two and take the
        # top most part which is the command space. Return the top at index[0].
        return np.array_split(totalActiveGrid, 2)[0]

    def spatialTemporal(self):
        i = 0
        for layer in self.layerArray:
            #print "     Layer = %s" % i
            i += 1
            layer.timeStep = layer.timeStep+1
            ## Update the current layers input with the new input
            # This updates the spatial pooler
            layer.Overlap()
            layer.inhibition(layer.timeStep)
            layer.learning()
            # This Updates the temporal pooler
            layer.updateActiveState(layer.timeStep)
            layer.updatePredictiveState(layer.timeStep)
            layer.sequenceLearning(layer.timeStep)


class HTM:
    #@do_cprofile  # For profiling
    def __init__(self, input, params):

        # This class contains multiple HTM levels stacked on one another
        self.numLevels = params['HTM']['numLevels']   # The number of levels in the HTM network
        self.width = params['HTM']['columnArrayWidth']
        self.height = params['HTM']['columnArrayHeight']
        self.cellsPerColumn = params['HTM']['cellsPerColumn']
        self.regionArray = np.array([], dtype=object)

        # Get just the parameters for the HTMRegion
        # Note the params comes in a list of dictionaries, one for each region.
        htmRegionParams = params['HTM']['HTMRegions']
        bottomRegionsParams = htmRegionParams[0]

        # Create a top level feedback input to store a command for the top level.
        # Used to direct the HTM network.
        self.topLevelFeedback = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                                         for j in range(self.height)])

        ### Setup the inputs and outputs between levels
        # Each regions input needs to make room for the command
        # feedback from the higher level.
        commandFeedback = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                                    for j in range(self.height)])
        # The lowest region receives the new input.
        newInput = SDRFunct.joinInputArrays(commandFeedback, input)
        # Setup the new htm region with the input and size parameters defined.
        self.regionArray = np.append(self.regionArray,
                                     HTMRegion(newInput,
                                               self.width,
                                               self.height,
                                               self.cellsPerColumn,
                                               bottomRegionsParams)
                                     )
        # The higher levels get inputs from the lower levels.
        #highestLevel = self.numLevels-1
        for i in range(1, self.numLevels):
            # Try to get the parameters for this region else use the
            # last specified params from the highest region.
            if len(htmRegionParams) >= i+1:
                regionsParam = htmRegionParams[i]
            else:
                regionsParam = htmRegionParams[-1]
            lowerOutput = self.regionArray[i-1].regionOutput()
            newInput = SDRFunct.joinInputArrays(commandFeedback, lowerOutput)
            self.regionArray = np.append(self.regionArray,
                                         HTMRegion(newInput,
                                                   self.width,
                                                   self.height,
                                                   self.cellsPerColumn,
                                                   regionsParam)
                                         )

        # create a place to store layers so they can be reverted.
        #self.HTMOriginal = copy.deepcopy(self.regionArray)
        self.HTMOriginal = None

    def updateTopLevelFb(self, newCommand):
        # Update the top level feedback command with a new one.
        self.topLevelFeedback = newCommand

    def saveRegions(self):
        # Save the HTM so it can be reloaded.
        print "\n    SAVE COMMAND SYN "
        self.HTMOriginal = copy.deepcopy(self.regionArray)

    def loadRegions(self):
        # Save the synases for the command area so they can be reloaded.
        if self.HTMOriginal is not None:
            print "\n    LOAD COMMAND SYN "
            self.regionArray = self.HTMOriginal
            # Need create a new deepcopy of the original
            self.HTMOriginal = copy.deepcopy(self.regionArray)
        # return the pointer to the HTM so the GUI can use it to point
        # to the correct object.
        return self.regionArray

    def updateHTMInput(self, input):
        # Update the input and outputs of the levels.
        # Level 0 receives the new input. The higher levels
        # receive inputs from the lower levels outputs

        # The input must also include the
        # command feedback from the higher layers.
        commFeedbackLev1 = np.array([])

        ### LEVEL 0 Update

        # The lowest levels lowest layer gets this new input.
        # All other levels and layers get inputs from lower levels and layers.
        if self.numLevels > 1:
            commFeedbackLev1 = self.levelOutput(1)
        # If there is only one level then the
        # thalamus input is added to the input.
        if self.numLevels == 1:
            # This is the highest level so get the
            # top level feedback command.
            commFeedbackLev1 = self.topLevelFeedback
        newInput = SDRFunct.joinInputArrays(commFeedbackLev1, input)
        self.regionArray[0].updateRegionInput(newInput)

        ### HIGHER LEVELS UPDATE

        # Update each levels input. Combine the command feedback to the input.
        for i in range(1, self.numLevels):
            commFeedbackLevN = np.array([])
            lowerLevel = i-1
            higherLevel = i+1
            # Set the output of the lower level
            highestLayer = self.regionArray[lowerLevel].numLayers - 1
            lowerLevelOutput = self.regionArray[lowerLevel].layerOutput(highestLayer)
            # Check to make sure this isn't the highest level
            if higherLevel < self.numLevels:
                # Get the feedback command from the higher level
                commFeedbackLevN = self.levelOutput(higherLevel)
            else:
                # This is the highest level so get the
                # top level feedback command.
                commFeedbackLevN = self.topLevelFeedback

            # Update the newInput for the current level in the HTM
            newInput = SDRFunct.joinInputArrays(commFeedbackLevN, lowerLevelOutput)
            self.regionArray[i].updateRegionInput(newInput)

    def levelOutput(self, level):
        # Return the output from the desired level.
        # The output will be from the highest layer in the level.
        highestLayer = self.regionArray[level].numLayers-1
        return self.regionArray[level].layerOutput(highestLayer)
        #return self.regionArray[level].regionOutput()

    def updateAllThalamus(self):
        # Update all the thalaums classes in each region
        for i in range(self.numLevels):
            self.regionArray[i].updateThalamus()

    def rewardAllThalamus(self, reward):
        # Reward the thalamus classes in each region
        for i in range(self.numLevels):
            self.regionArray[i].rewardThalamus(reward)

    #@do_cprofile  # For profiling
    def spatialTemporal(self, input):
        # Update the spatial and temporal pooler.
        # Find spatial and temporal patterns from the input.
        # This updates the columns and all their vertical
        # synapses as well as the cells and the horizontal Synapses.
        # Update the current levels input with the new input
        self.updateHTMInput(input)
        i = 0
        for level in self.regionArray:
            #print "Level = %s" % i
            i += 1
            level.spatialTemporal()




