# Title: HTM
# Description: git managed development of a HTM network
# Author: Calum Meiklejohn
# Development phase: alpha

#import HTM_draw
#import pygame
import numpy as np
import random
import math
#import pprint
import copy

##Struct = {'field1': 'some val', 'field2': 'some val'}
##myStruct = { 'num': 1}

SegmentUpdate = {'index': '-1',
                 'activeSynapses': '0',
                 'sequenceSegment': 0,
                 'createdAtTime': 0}


class Synapse:
    def __init__(self, input, pos_x, pos_y, cellIndex):
            # cell is -1 if the synapse connects the HTM layers input.
            # Otherwise it is a horizontal connection to the cell
            # index self.cell in the column at self.pos_x self.pos_y
            self.cell = cellIndex
            self.pos_x = pos_x
            # The end of the synapse. The start is at a column or cells position
            self.pos_y = pos_y
            self.permanence = 0.38
            #If the permanence value for a synapse is greater than this
            #value, it is said to be connected.
            self.connectPermanence = 0.3


class Segment:
    def __init__(self):
        self.predict = False
        self.index = -1
        self.sequenceSegment = 0
        # Stores the last time step that this segment was predicting activity
        # Stores the synapses that have been created and
        #have a larger permenence than 0.0
        self.synapses = np.array([], dtype=object)


class Cell:
    def __init__(self):
        # dendrite segments
        #self.numInitSegments = 1    # Must be greater then zero
        self.score = 0     # The current score for the cell.
        self.segments = np.array([], dtype=object)
        #for i in range(self.numInitSegments):
        #    self.segments = np.hstack((self.segments,[Segment()]))
        # Create a dictionary to store the segmentUpdate structures
        self.segmentUpdateList = []
        # Segment update stucture holds the updates for the cell.
        #These updates are made later.
        self.segmentUpdate = {'index': -1,
                              'activeSynapses': np.array([], dtype=object),
                              'newSynapses': np.array([], dtype=object),
                              'sequenceSegment': 0,
                              'createdAtTime': 0}
        #for i in range(self.numInitSegments):
        #    self.segmentUpdateList.append(self.segmentUpdate.copy())
        #print self.segmentUpdateList
##        # State of the cell
##        self.active = False
##        self.predict = False


class Column:
    def __init__(self, length, pos_x, pos_y, input):
        self.cells = np.array([], dtype=object)
        for i in range(length):
            self.cells = np.hstack((self.cells, [Cell()]))
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.overlap = 0.0
        self.minOverlap = 4
        self.boost = 1
        # The max distance a column can inhibit another column.
        #This parameters value is automatically reset.
        self.inhibitionRadius = 1
        # The max distance that Synapses can be made at
        self.potentialRadius = 1
        self.permanenceInc = 0.1
        self.permanenceDec = 0.02
        self.minDutyCycle = 0.01   # The minimum firing rate of the column
        self.activeDutyCycleArray = np.array([0])
        # Keeps track of when the column was active.
        # All columns start as active. It stores the
        #numInhibition time when the column was active
        self.activeDutyCycle = 0.0  # the firing rate of the column
        self.activeState = False
        # The rate at which the overlap is larger then the min overlap
        self.overlapDutyCycle = 0.0
        # Keeps track of when the colums overlap was larger then the minoverlap
        self.overlapDutyCycleArray = np.array([0])
        self.boostStep = 0.1
        # This determines how many previous timeSteps are stored in
        #actve predictive and learn state arrays.
        self.historyLength = 2
        self.highestScoredCell = None

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
    # POSSIBLY MOVE THESE FUNCTIONS TO THE HTMLayer CLASS?

    def updateConnectedSynapses(self):
        self.connectedSynapses = np.array([], dtype=object)
        for i in range(len(self.potentialSynapses)):
            if self.potentialSynapses[i].permanence > self.potentialSynapses[i].connectPermanence:
                self.connectedSynapses = np.append(self.connectedSynapses, self.potentialSynapses[i])
##    def input(self,input):
##        # Update the selected synapses inputs
##        for i in range(len(self.potentialSynapses)):
##            self.potentialSynapses[i].updateInput(input)

    def updateBoost(self):
        if self.activeDutyCycle < self.minDutyCycle:
            self.boost = self.boost+self.boostStep
        else:
            #print "activeDutyCycle %s > minDutyCycle %s"
            # %(self.activeDutyCycle,self.minDutyCycle)
            self.boost = 1.0
        #print self.boost

    #def updateArray(self,timeStep,array):
        ## This function will be used if activeArray ends up storing more
        # than just the last active time.
        # This function is used to update the predict, active and
        #learn state arrays.
        # It adds the new time to the start of the list and deletes the
        # last item in the list.
        # This way the newest times are always at the start of the list.
        #array.insert(0,timeStep)
        #del(array[len(array)-1])
        #return array


class HTMLayer:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn):
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
        self.desiredLocalActivity = 1
        self.cellsPerColumn = cellsPerColumn
        self.connectPermanence = 0.3
        # Should be smaller than activationThreshold
        self.minThreshold = 4
        # The minimum score needed by a cell to be added
        # to the alternative sequence.
        self.minScoreThreshold = 3
        # This limits the activeSynapse array to this length. Should be renamed
        self.newSynapseCount = 10
        # More than this many synapses on a segment must be active for
        # the segment to be active
        self.activationThreshold = 5
        self.dutyCycleAverageLength = 1000
        self.timeStep = 0
        self.output = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        self.activeColumns = np.array([], dtype=object)
        self.averageReceptiveFeildSizeArray = np.array([])
        # Create the array storing the columns
        self.columns = np.array([[Column(self.cellsPerColumn, i, j, input) for
                                i in range(columnArrayWidth)] for
                                j in range(columnArrayHeight)], dtype=object)
        # Initialise the columns potential synapses.
        # Work out the potential feedforward connections each column could make to the input.
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                self.updatePotentialSynapses(c)

    def activeCellGrid(self):
        ############### TODO ############
        # Return a grid representing the cells in the columns which are active but
        # not bursting. Cells in a column are placed in adjacent grid cells right of each other.
        # Eg. A HTM layer with 10 rows, 5 columns and 3 cells per column would produce an
        # activeCellGrid of 10*3 = 15 columns and 10 rows.
        output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                          for j in range(self.height)])
        for k in range(len(self.columns)):
            for p in range(len(self.columns[k])):
                c = self.columns[k][p]
                for i in range(len(c.cells)):
                    if c.activeStateArray[i] == self.timeStep:
                        output[k][p*self.cellsPerColumn+i] = 1
        print "output = ", output
        return output

    def updateOutput(self):
        # Update the output array.
        # Initialise all outputs as zero first then set the active columns as 1.
        for i in range(len(self.output)):
            for j in range(len(self.output[i])):
                self.output[i][j] = 0
        for i in range(len(self.activeColumns)):
            x = self.activeColumns[i].pos_x
            y = self.activeColumns[i].pos_y
            self.output[y][x] = 1

    def updatePotentialSynapses(self, c):
        # Update the locations of the potential synapses for column c.
        # If the input is larger than the number of columns then
        # the columns are evenly spaced out over the input.
        inputHeigth = len(self.Input)
        inputWidth = len(self.Input[0])
        columnHeight = len(self.columns)
        columnWidth = len(self.columns[0])
        # Calculate the ratio between columns and the input space.
        colInputRatioHeight = float(inputHeigth / columnHeight)
        colInputRatioWidth = float(inputWidth / columnWidth)
        #print "Column to input height ratio = %s" % colInputRatioHeight
        #print "Column to input width ratio = %s" % colInputRatioWidth

        # i is pos_y j is pos_x
        for y in range(int(c.pos_y-c.potentialRadius),
                       int(c.pos_y+c.potentialRadius)+1):
            # Cast the y position to an int so it matches up with a row number.
            inputPos_y = int(y*colInputRatioHeight)
            if inputPos_y >= 0 and inputPos_y < inputHeigth:
                for x in range(int(c.pos_x-c.potentialRadius),
                               int(c.pos_x+c.potentialRadius)+1):
                    # Cast the x position to an int so it matches up with a column number.
                    inputPos_x = int(x*colInputRatioWidth)
                    if inputPos_x >= 0 and inputPos_x < inputWidth:
                        # Create a Synapse pointing to the HTM layers input
                        #so the synapse cellIndex is -1
                        c.potentialSynapses = np.append(c.potentialSynapses,
                                                        [Synapse(self.Input, inputPos_x, inputPos_y, -1)])

    def neighbours(self, c):
        # returns a list of the columns that are within the inhibitionRadius of c
        closeColumns = np.array([], dtype=object)
        # Add one to the c.pos_y+c.inhibitionRadius because for example range(0,2)=(0,1)
        for i in range(int(c.pos_y-c.inhibitionRadius), int(c.pos_y+c.inhibitionRadius)+1):
            if i >= 0 and i < (len(self.columns)):
                for j in range(int(c.pos_x-c.inhibitionRadius), int(c.pos_x+c.inhibitionRadius)+1):
                    if j >= 0 and j < (len(self.columns[0])):
                        closeColumns = np.append(closeColumns, self.columns[i][j])
        return closeColumns

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

    def boostFunction(c):
        pass

    def kthScore(self, cols, kth):
        if len(cols) > 0 and kth > 0 and kth < (len(cols)-1):
            orderedScore = np.array(cols[0].overlap)
            #print cols[0].overlap
            for i in range(1, len(cols)):
            #Add the overlap values to a single list
                orderedScore = np.append(orderedScore, [cols[i].overlap])
            orderedScore = np.sort(orderedScore)
            #print orderedScore
            return orderedScore[-kth]       # Minus since list starts at lowest
        return 0

    def averageReceptiveFeildSize(self):
        self.averageReceptiveFeildSizeArray = np.array([])
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                self.averageReceptiveFeildSizeArray = np.append(self.averageReceptiveFeildSizeArray,
                                                                len(c.connectedSynapses))
        #print np.average(self.averageReceptiveFeildSizeArray)
        #Returns the radius of the average receptive feild size
        return int(math.sqrt(np.average(self.averageReceptiveFeildSizeArray))/2)

    def updateActiveDutyCycle(self, c):
        if c.activeState is True:
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

    def Cell(c, i):
        pass

    def activeColumns(t):
        pass

    def deleteEmptySegments(self, c, i):
        # Delete the segments that have no synapses in them.
        # This should only be done before or after learning
        deleteEmptySegments = []
        # This is a list of the indicies of the segments that will be deleted
        for s in range(len(c.cells[i].segments)):
            if len(c.cells[i].segments[s].synapses) == 0:
                deleteEmptySegments.append(s)
        if len(deleteEmptySegments) > 0:  # Used for printing only
            print "Deleted %s segments from x, y, i=%s, %s, %s segindex = %s"%(len(deleteEmptySegments), c.pos_x,c.pos_y,i,deleteEmptySegments)
        c.cells[i].segments = np.delete(c.cells[i].segments, deleteEmptySegments)

    def deleteWeakSynapses(self, c, i, segIndex):
        # Delete the synapses that have a permanence
        # value that is too low form the segment.
        deleteActiveSynapses = []
        # This is a list of the indicies of the
        # active synapses that will be deleted
        for k in range(len(c.cells[i].segments[segIndex].synapses)):
            syn = c.cells[i].segments[segIndex].synapses[k]
            if syn.permanence < syn.connectPermanence:
                deleteActiveSynapses.append(k)
        c.cells[i].segments[segIndex].synapses = np.delete(c.cells[i].segments[segIndex].synapses, deleteActiveSynapses)
        #print "     deleted %s number of synapses"%(len(deleteActiveSynapses))

    def columnActiveAdd(self, c, timeStep):
        # We add the new time to the start of the array
        # then delete the time at the end of the array.
        # All the times should be in order from highest
        # (most recent) to lowest (oldest).
        newArray = np.insert(c.columnActive, 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.columnActive = newArray

    def activeStateAdd(self, c, i, timeStep):
        # We add the new time to the start of
        # the array then delete the time at the end of the array.
        # All the times should be in order from highest
        # (most recent) to lowest (oldest).
        newArray = np.insert(c.activeStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.activeStateArray[i] = newArray

    def predictiveStateAdd(self, c, i, timeStep):
        # We add the new time to the start of the array
        # then delete the time at the end of the array.
        # All the times should be in order from
        # highest (most recent) to lowest (oldest).
        newArray = np.insert(c.predictiveStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.predictiveStateArray[i] = newArray

    def learnStateAdd(self, c, i, timeStep):
        # We add the new time to the start of the
        # array then delete the time at the end of the array.
        # All the times should be in order from highest
        # (most recent) to lowest (oldest).
        newArray = np.insert(c.learnStateArray[i], 0, timeStep)
        newArray = np.delete(newArray, len(newArray)-1)
        c.learnStateArray[i] = newArray

    def columnActiveState(self, c, timeStep):
        # Search the history of the columnActive to find if the
        # column was predicting at time timeStep
        for j in range(len(c.columnActive)):
            if c.columnActive[j] == timeStep:
                return True
        return False

    def activeState(self, c, i, timeStep):
        # Search the history of the activeStateArray to find if the
        # cell was predicting at time timeStep
        for j in range(len(c.activeStateArray[i])):
            if c.activeStateArray[i, j] == timeStep:
                return True
        return False

    def predictiveState(self, c, i, timeStep):
        # Search the history of the predictiveStateArray to find if the
        # cell was active at time timeStep
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

    def randomActiveSynapses(self, c, i, s, timeStep):
        # Randomly add self.newSynapseCount-len(synapses) number of Synapses
        # that connect with cells that are active
        #print "randomActiveSynapses time = %s"%timeStep
        synapseList = np.array([], dtype=object)
        # A list of all potential synapses that are active
        for l in range(len(self.columns)):
        # Can't use c since c already represents a column
            for m in self.columns[l]:
                for j in range(len(m.learnStateArray)):
                    if self.learnState(m, j, timeStep) is True:
                        #print "time = %s synapse ends at
                        # active cell x,y,i = %s,%s,%s"%(timeStep,m.pos_x,m.pos_y,j)
                        synapseList = np.append(synapseList, Synapse(0, m.pos_x, m.pos_y, j))
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
            return np.array([])
        return np.array(random.sample(synapseList, numNewSynapses))

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
            if s.synapses[i].permanence > s.synapses[i].connectPermanence:
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
                    # Must be larger than or equal to
                    #otherwise cell 0 segment 0 will never
                    #be chosen as the best cell
                    if (self.segmentNumSynapsesActive(c.cells[i].segments[h], timeStep, True) >= self.segmentNumSynapsesActive(c.cells[bestCell].segments[bestSegment], timeStep, True)):
                        # Remeber the best cell and segment
                        bestCell = i
                        bestSegment = h
                        bestCellFound = True
        if bestCellFound is True:
            #print "returned from GETBESTMATCHINGCELL the cell
            #i=%s with the best segment s=%s"%(bestCell,bestSegment)
            return (bestCell, bestSegment)
        else:
            # Return the first segment from the cell with the fewest segments
            #print "returned from getBestMatchingCell cell i=%s
            #with the fewest number of segments
            #num=%s"%(fewestSegments,len(c.cells[fewestSegments].segments))
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
        newSegmentUpdate = {'index': s, 'activeSynapses': np.array([], dtype=object),
                            'newSynapses': np.array([], dtype=object), 'sequenceSegment': 0,
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
                            newSegmentUpdate['activeSynapses'] = np.append(newSegmentUpdate['activeSynapses'],
                                                                           c.cells[i].segments[s].synapses[k])
        if newSynapses is True:
            # Add new synapses that have an active end.
            # We add them  to the segmentUpdate structure. They are added
            # To the actual segment later during learning when the cell is
            # in a learn state.
            newSegmentUpdate['newSynapses'] = np.append(newSegmentUpdate['newSynapses'],
                                                        self.randomActiveSynapses(c, i, s, timeStep))
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
            segIndex = c.cells[i].segmentUpdateList[j]['index']
            #print "     segIndex = %s"%segIndex
            # If the segment exists
            if segIndex > -1:
                #print "     adapted x,y,cell,segment=
                #%s,%s,%s,%s"%(c.pos_x,c.pos_y,i,c.cells
                    #[i].segmentUpdateList[j]['index'])
                for s in c.cells[i].segmentUpdateList[j]['activeSynapses']:
                    # For each synapse in the segments activeSynapse
                    #list increment or
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
                c.cells[i].segments[segIndex].synapses = np.append(c.cells[i].segments[segIndex].synapses,
                                                                   c.cells[i].segmentUpdateList[j]['newSynapses'])
                # Delete synapses that have low permanence values.
                self.deleteWeakSynapses(c, i, segIndex)
            # If the segment is new (the segIndex = -1) add it to the cell
            else:
                newSegment = Segment()
                newSegment.synapses = c.cells[i].segmentUpdateList[j]['newSynapses']
                c.cells[i].segments = np.append(c.cells[i].segments, newSegment)
                #print "     new segment added for x,y,cell,seg =
                #%s,%s,%s,%s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments)-1)
                #for s in c.cells[i].segments[len(c.cells[i].
                    #segments)-1].synapses:
                # Used for printing only
                #    print "         synapse ends at
                #x,y=%s,%s"%(s.pos_x,s.pos_y)
        c.cells[i].segmentUpdateList = []
        # Delete the list as the updates have been added.

    def updateInput(self, input):
        self.Input = input

    def Overlap(self):
        # Phase one for the spatial pooler
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.overlap = 0.0
                c.updateConnectedSynapses()
                for s in c.connectedSynapses:
                    # Check if the input that this synapses
                    #is connected to is active.
                    #print "s.pos_y = %s s.pos_x = %s" % (s.pos_y, s.pos_x)
                    #print "input width = %s input height = %s" % (len(self.Input[0]), len(self.Input))
                    inputActive = self.Input[s.pos_y][s.pos_x]
                    c.overlap = c.overlap + inputActive
                if c.overlap < c.minOverlap:
                    c.overlap = 0.0
                else:
                    c.overlap = c.overlap*c.boost
                    self.updateOverlapDutyCycle(c)
                #print "%d %d %d" %(c.overlap,c.minOverlap,c.boost)

    def inhibition(self, timeStep):
        # Phase two for the spatial pooler
        self.activeColumns = np.array([], dtype=object)
        #print "actve cols before %s" %self.activeColumns
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.activeState = False
                if c.overlap > 0:
                    minLocalActivity = self.kthScore(self.neighbours(c), self.desiredLocalActivity)
                    #print "current column = (%s,%s)"%(c.pos_x,c.pos_y)
                    if c.overlap > minLocalActivity:
                        self.activeColumns = np.append(self.activeColumns, c)
                        c.activeState = True
                        self.columnActiveAdd(c, timeStep)
                        #print "ACTIVE COLUMN x,y = %s,%s overlap
                        #= %d min = %d" %(c.pos_x,c.pos_y,
                            #c.overlap,minLocalActivity)
                    if c.overlap == minLocalActivity:
                        # Check the active columns array and see how many columns
                        # near the current one are already active.
                        numNeighbours = 0
                        for d in self.activeColumns:
                            if self.areNeighbours(c, d) is True:
                                numNeighbours += 1
                        # if less then the desired local activity have been set as active
                        # then activate this column as well
                        if numNeighbours < self.desiredLocalActivity:
                            #print "Activated column numNeighbours = %s" % numNeighbours
                            self.activeColumns = np.append(self.activeColumns, c)
                            c.activeState = True
                            self.columnActiveAdd(c, timeStep)
                self.updateActiveDutyCycle(c)
                # Update the active duty cycle variable of every column

    def learning(self):
        # Phase three for the spatial pooler
        for c in self.activeColumns:
            for s in c.potentialSynapses:
                # Check if the input that this
                #synapses is connected to is active.
                inputActive = self.Input[s.pos_y][s.pos_x]
                if inputActive == 1:
                #Only handles binary input sources
                    s.permanence += c.permanenceInc
                    s.permanence = min(1.0, s.permanence)
                else:
                    s.permanence -= c.permanenceDec
                    s.permanence = max(0.0, s.permanence)
        average = self.averageReceptiveFeildSize()
        #Find the average of the receptive feild sizes just once
        #print "inhibition radius = %s" %average
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.minDutyCycle = 0.01*self.maxDutyCycle(self.neighbours(c))
                c.updateBoost()
                c.inhibitionRadius = average
                # Set to the average of the receptive feild sizes.
                #All columns have the same inhibition radius
                if c.overlapDutyCycle < c.minDutyCycle:
                    self.increasePermanence(c, 0.1*self.connectPermanence)
        self.updateOutput()

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
        This active cell means it has been part of an alternative sequence that that was also
        being predicted by HTM layer.
        """
        # First reset the active cells calculated from the previous time step.
        print "       1st TEMPORAL FUNCTION"
        # This is different to CLA paper.
        # First we calculate the score for each cell in the active column
        for c in self.activeColumns:
            #print "\n ACTIVE COLUMN x,y = %s,%s time =
            #%s"%(c.pos_x,c.pos_y,timeStep)
            #print "columnActive =",c.columnActive
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
        # According to the CLA paper
        for c in self.activeColumns:
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
                    print"best SCORE active x, y, i = %s, %s, %s score = %s"%(c.pos_x, c.pos_y,c.highestScoredCell, c.cells[c.highestScoredCell].score)
                    buPredicted = True
                    self.activeStateAdd(c, c.highestScoredCell, timeStep)
                    lcChosen = True
                    self.learnStateAdd(c, c.highestScoredCell, timeStep)
                    # Add a new Segment
                    sUpdate = self.getSegmentActiveSynapses(c, c.highestScoredCell, timeStep-1, -1, True)
                    sUpdate['sequenceSegment'] = timeStep
                    c.cells[c.highestScoredCell].segmentUpdateList.append(sUpdate)
                if lcChosen is False and c.cells[c.highestScoredCell].score >= self.minScoreThreshold:
                    print"best SCORE learn x,y,i = %s,%s,%s score = %s" % (c.pos_x, c.pos_y, c.highestScoredCell, c.cells[c.highestScoredCell].score)
                    lcChosen = True
                    self.learnStateAdd(c, c.highestScoredCell, timeStep)
                    # Add a new Segment
                    sUpdate = self.getSegmentActiveSynapses(c, c.highestScoredCell, timeStep-1, -1, True)
                    sUpdate['sequenceSegment'] = timeStep
                    c.cells[c.highestScoredCell].segmentUpdateList.append(sUpdate)

            # According to the CLA paper
            if buPredicted is False:
                #print "No cell in this column predicted"
                for i in range(self.cellsPerColumn):
                    self.activeStateAdd(c, i, timeStep)
            if lcChosen is False:
                #print "lcChosen Getting the best matching
                #cell to set as learning cell"
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
        print "\n       2nd TEMPORAL FUNCTION "
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

    def temporalLearning(self, timeStep):
        # Third function called for the sequence pooler.
        # The update structures are implemented on the cells
        print "\n       3rd TEMPORAL FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    #print "predictiveStateArray for x,y,i =
                    #%s,%s,%s is latest time = %s"%(c.pos_x,c.pos_y,i,
                        #c.predictiveStateArray[i,0])
                    if self.learnState(c, i, timeStep) is True:
                        #print "learn state for x,y,cell =
                        #%s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, True)
                    # Trying a different method to the CLA white pages
                    #if self.activeState(c,i,timeStep) ==
                    #False and self.predictiveState(c,i,timeStep-1) is True:
                    # Same method as the CLA white pages.
                    if (self.predictiveState(c, i, timeStep-1) is True
                            and self.activeState(c, i, timeStep) is False):
                        #print "INCORRECT predictive
                        #state for x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, False)
                    # After the learning delete any segments
                    #that have zero synapses in them.
                    # This must be done after learning since
                    #during learning the index of the segment
                    # is used to identify each segment and this
                    #changes when segments are deleted.
                    self.deleteEmptySegments(c, i)


class HTMRegion:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn):
        self.quit = False
        # The class contains multiple HTM layers stacked on one another
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.cellsPerColumn = cellsPerColumn

        self.numLayers = 1  # The number of HTM layer that make up a region.

        self.layerArray = np.array([], dtype=object)
        # Set up the inputs to the HTM layers.
        # Layer 0 gets the new input.
        # The higher layers receive the lower layers output.
        self.layerArray = np.append(self.layerArray, HTMLayer(input, self.width,
                                                              self.height, self.cellsPerColumn))
        for i in range(1, self.numLayers):
            lowerOutput = self.layerArray[i-1].output
            self.layerArray = np.append(self.layerArray,
                                        HTMLayer(lowerOutput, self.width,
                                        self.height, self.cellsPerColumn))

    def updateRegionInput(self, input):
        # Update the input and outputs of the layers.
        # Layer 0 receives the new input. The higher layers
        # receive inputs from the lower layer outputs
        self.layerArray[0].updateInput(input)
        for i in range(1, self.numLayers):
            self.layerArray[i].updateInput(self.layerArray[i-1].output)

    def regionOutput(self):
        # Return the regions output from its highest layer.
        highestLayer = self.numLayers-1
        return self.layerArray[highestLayer].output

    def regionCommandOutput(self):
        # Return the regions command output from its command layer (the highest layer).
        # The command output is the grid of the active non bursted cells.
        highestLayer = self.numLayers-1
        return self.layerArray[highestLayer].activeCellGrid()

    def spatialTemporal(self):
        i = 0
        for layer in self.layerArray:
            print "     Layer = %s" % i
            i += 1
            layer.timeStep = layer.timeStep+1
            ## Update the current layers input with the new input
            ##self.layerArray[layerNum].updateInput(input)
            # This updates the spatial pooler
            layer.Overlap()
            layer.inhibition(layer.timeStep)
            layer.learning()
            # This Updates the temporal pooler
            layer.updateActiveState(layer.timeStep)
            layer.updatePredictiveState(layer.timeStep)
            layer.temporalLearning(layer.timeStep)


class HTM:
    def __init__(self, numLevels, input, columnArrayWidth,
                 columnArrayHeight, cellsPerColumn):
        self.quit = False
        # The class contains multiple HTM levels stacked on one another
        self.numLevels = numLevels   # The number of levels in the HTM network
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.cellsPerColumn = cellsPerColumn

        self.HTMRegionArray = np.array([], dtype=object)
        # The lowest region
        self.HTMRegionArray = np.append(self.HTMRegionArray,
                                        HTMRegion(input, self.width, self.height,
                                                  self.cellsPerColumn))
        # The higher levels get inputs from the lower levels.
        for i in range(1, numLevels):
            lowerOutput = self.HTMRegionArray[i-1].regionOutput()
            self.HTMRegionArray = np.append(self.HTMRegionArray,
                                            HTMRegion(lowerOutput, self.width, self.height,
                                                      self.cellsPerColumn))
        # create a place to store layers so they can be reverted.
        self.HTMOriginal = copy.deepcopy(self.HTMRegionArray)

    def saveRegions(self):
        # Save the HTM so it can be reloaded.
        print "\n    SAVE COMMAND SYN "
        self.HTMOriginal = copy.deepcopy(self.HTMRegionArray)

    def loadRegions(self):
        # Save the synases for the command area so they can be reloaded.
        print "\n    LOAD COMMAND SYN "
        self.HTMRegionArray = self.HTMOriginal
        # Need create a new deepcopy of the original
        self.HTMOriginal = copy.deepcopy(self.HTMRegionArray)
        # return the pointer to the HTM so the GUI can use it to point
        # to the correct object.
        return self.HTMRegionArray

    def updateHTMInput(self, input):
        # Update the input and outputs of the levels.
        # Level 0 receives the new input. The higher levels
        # receive inputs from the lower levels outputs
        self.HTMRegionArray[0].updateRegionInput(input)
        for i in range(1, self.numLevels):
            lowerLevel = i-1
            lowerLevelOutput = self.HTMRegionArray[lowerLevel].regionOutput()
            self.HTMRegionArray[i].updateRegionInput(lowerLevelOutput)

    def levelCommandOutput(self, level):
        # Return the command output of the desired level.
        return self.HTMRegionArray[0].regionOutput()

    def spatialTemporal(self, input):
        # Update the spatial and temporal pooler.
        # Find spatial and temporal patterns from the input.
        # This updates the columns and all there vertical
        # synapses as well as the cells and the horizontal Synapses.
        # Update the current levels input with the new input
        self.updateHTMInput(input)
        i = 0
        for level in self.HTMRegionArray:
            print "Level = %s" % i
            i += 1
            level.spatialTemporal()




