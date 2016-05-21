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
from reinforcement_learning import Thalamus

from HTM_calc import theano_temporal as temporal
from HTM_calc import theano_overlap as overlap
#from HTM_calc import theano_inhibition as inhibition
from HTM_calc import np_inhibition as inhibition
#from HTM_calc import theano_learning as learning
from HTM_calc import np_learning as learning

from HTM_calc import np_activeCells as activeCells
from HTM_calc import np_predictCells as predictCells



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
    def __init__(self, pos_x, pos_y, cellIndex, permanence):
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
        self.boost = params['boost']
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
        # The inhibition distance width and height. How far away a column can inhibit another
        # column.
        self.inhibitionHeight = params['inhibitionHeight']
        self.inhibitionWidth = params['inhibitionWidth']
        # If true then all the potential synapses for a column are centered
        # around the columns position else they are to the right of the columns pos.
        self.centerPotSynapses = params['centerPotSynapses']
        self.cellsPerColumn = cellsPerColumn
        # If the permanence value for a synapse is greater than this
        # value, it is said to be connected.
        self.connectPermanence = params['connectPermanence']
        # Should be smaller than activationThreshold.
        # More then this many synapses in a segment must be active for the segment
        # to be considered for an alternative sequence (to increment a cells score).
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
        # The starting permanence of new column synapses (spatial pooler synapses).
        # This is used to create new synapses.
        self.colSynPermanence = params['colSynPermanence']
        # The starting permanence of new cell synapses (sequence pooler synapses).
        # This is used to create new synapses.
        self.cellSynPermanence = params['cellSynPermanence']

        # Create matrix group variables. These variables store info about all
        # the columns, cells and synpases. This is done to improve the performance
        # since operations are just matrix manipulations.
        # These parameters come from the column class.
        # Just take the parameters for the first column.

        self.potentialWidth = params['potentialWidth']
        self.potentialHeight = params['potentialHeight']
        self.minOverlap = params['minOverlap']
        self.spatialPermanenceInc = params['spatialPermanenceInc']
        self.spatialPermanenceDec = params['spatialPermanenceDec']
        self.maxNumTempPoolPatterns = params['maxNumTempPoolPatterns']
        # Already active columns have their spatial synapses decremented by a different value in the spatial pooler.
        self.activeColPermanenceDec = (float(self.spatialPermanenceInc) /
                                       float(self.maxNumTempPoolPatterns))
        self.permanenceInc = params['permanenceInc']
        self.permanenceDec = params['permanenceDec']
        self.inputHeight = len(self.Input)
        self.inputWidth = len(self.Input[0])
        self.numPotSyn = self.potentialWidth * self.potentialHeight
        self.numColumns = self.height * self.width
        # Setup a matrix where each row represents a list of a columns
        # potential synapse permanence values
        self.colPotSynPerm = np.array([[self.colSynPermanence for i in range(self.numPotSyn)]
                                      for j in range(self.numColumns)])
        # Setup a matrix where each position represents a columns overlap.
        self.colOverlaps = np.empty([self.height, self.width])
        # Setup a matrix where each row represents a columns input values from its potential synapses.
        self.colPotInputs = np.empty([self.numColumns, self.numPotSyn])
        # Setup a matrix where each element represents the timestep when a column
        # was active but not bursting last. Each position in the first dimension
        # represents a column. The matrix stores the last two times the column
        # was active but not bursting. The latest timeStep is stored in the first position.
        # eg. self.colActNotBurstTimes[41][0] stores the latest time that column 42 was active
        # but not bursting. self.colActNotBurstTimes[41][1] stores the second last time that column
        # 42 was active but not bursting. The third place is a temporary position used to update
        # the other two positions.
        self.colActNotBurstTimes = np.zeros((self.numColumns, 3))
        self.tempTimeCheck = 0
        # Setup a vector where each element represents a timeStep when a column
        # should stop temporal pooling.
        self.colStopTempAtTime = np.zeros(self.numColumns)
        # Setup a vector where each element represents if a column is active 1 or not 0
        self.colActive = np.zeros(self.numColumns)

        # The timeSteps when cells where active last. This is a 3D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timestep when this cell was active last.
        self.activeCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)]
                                        for y in range(self.numColumns)])

        # The timeSteps when cells where in the predicitive state last. This is a 3D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timestep when this cell was predicitng last.
        self.predictCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)]
                                         for y in range(self.numColumns)])

        # Create the distalSynapse 5d tensor holding the information of the distal synapses.
        # The first dimension stores the columns, the 2nd is the cells
        # in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
        # segment and the 5th stores the end connection of the synapse
        # (column number, cell number, permanence). This tensor has a size of
        # numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
        # It does not change size. Its size is fixed when this class is constructed.
        self.distalSynapses = np.zeros((self.numColumns,
                                        self.cellsPerColumn,
                                        self.maxNumSegments,
                                        self.newSynapseCount, 3))
        # activeSeg is a 3D tensor. The first dimension is the columns, the second the
        # cells and the 3rd is the segment in the cells. For each segment a timeStep is stored
        # indicating when the segment was last in an active state. This means it was
        # predicting that the cell would become active in the next timeStep.
        # This is what the CLA paper calls a "SEQUENCE SEGMENT".
        self.activeSeg = np.zeros((self.numColumns, self.cellsPerColumn, self.maxNumSegments))
        # predictiveCells is a 3D tensor. The first dimension stores the columns the second
        # is the cells in the columns. Each cell stores the last two timeSteps when
        # the cell was in a predictiveState. It must have the dimesions of
        # self.numColumns * self.cellsPerColumn * 2.
        self.predictiveCells = np.zeros((self.numColumns,
                                        self.cellsPerColumn, 2))

        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # This tensor stores segment update info from the active Cells calculator.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdateActive = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        # This tensor stores segment update info from the active Cells calculator.
        self.segActiveSynActive = np.array([[[-1 for z in range(self.maxSynPerSeg)]
                                            for x in range(self.cellsPerColumn)]
                                            for y in range(self.numColumns)])
        # A 2D tensor "segIndNewSyn" for each cell holds [segIndex] indicating which segment new
        # synapses should be created for. If the index is -1 don't create any new synapses.
        # This tensor stores segment update info from the active Cells calculator.
        self.segIndNewSynActive = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 4D tensor "segNewSyn" for each cell holds a synapse list [newSynapseList] of new synapses
        # that could possibly be created. Each position corresponds to a synapses in the segment
        # with the index stored in the segNewSynActive tensor.
        # Each place in the newSynapseList holds [columnIndex, cellIndex, permanence]
        # If permanence is -1 then this means don't create a new synapse for that synapse.
        # This tensor stores segment update info from the active Cells calculator.
        self.segNewSynActive = np.array([[[[-1, -1, 0.0] for z in range(self.maxSynPerSeg)]
                                        for x in range(self.cellsPerColumn)]
                                        for y in range(self.numColumns)])
        # A 2D tensor "segIndNewSynActive" for each cell holds [segIndex] indicating which segment new
        # synapses should be created for. If the index is -1 don't create any new synapses.
        # This tensor stores segment update info from the active Cells calculator.
        self.segIndNewSynActive = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])

        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # This tensor stores segment update info from the predict Cells calculator.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdatePredict = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        # This tensor stores segment update info from the predictive Cells calculator.
        self.segActiveSynPredict = np.array([[[-1 for z in range(self.maxSynPerSeg)]
                                            for x in range(self.cellsPerColumn)]
                                            for y in range(self.numColumns)])

        # Create the array storing the columns
        self.columns = np.array([[]], dtype=object)
        # Setup the columns array.
        self.setupColumns(params['Columns'])

        # Setup the theano classes used for calculating
        # spatial, temporal and sequence pooling.
        self.setupCalculators()

        # Initialise the columns potential synapses.
        # Work out the potential feedforward connections each column could make to the input.
        self.setupPotentialSynapses(self.inputWidth, self.inputHeight)

    def setupCalculators(self):
        # Setup the theano calculator classes used to calculate
        # efficiently the spatial, temporal and sequence pooling.

        self.overlapCalc = overlap.OverlapCalculator(self.potentialWidth,
                                                     self.potentialHeight,
                                                     self.width,
                                                     self.height,
                                                     self.inputWidth,
                                                     self.inputHeight,
                                                     self.centerPotSynapses,
                                                     self.connectPermanence,
                                                     self.minOverlap)

        self.inhibCalc = inhibition.inhibitionCalculator(self.width, self.height,
                                                         self.inhibitionWidth,
                                                         self.inhibitionHeight,
                                                         self.desiredLocalActivity,
                                                         self.minOverlap,
                                                         self.centerPotSynapses)

        self.learningCalc = learning.LearningCalculator(self.numColumns,
                                                        self.numPotSyn,
                                                        self.spatialPermanenceInc,
                                                        self.spatialPermanenceDec,
                                                        self.activeColPermanenceDec)

        self.activeCellsCalc = activeCells.activeCellsCalculator(self.numColumns,
                                                                 self.cellsPerColumn,
                                                                 self.maxNumSegments,
                                                                 self.newSynapseCount,
                                                                 self.minThreshold,
                                                                 self.minScoreThreshold,
                                                                 self.cellSynPermanence)

        self.predictCellsCalc = predictCells.predictCellsCalculator(self.numColumns,
                                                                    self.cellsPerColumn,
                                                                    self.maxNumSegments,
                                                                    self.newSynapseCount,
                                                                    self.connectPermanence)

        self.tempPoolCalc = temporal.TemporalPoolCalculator(self.potentialWidth,
                                                            self.potentialHeight,
                                                            self.minOverlap)



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

    def columnActiveNotBursting(self, col, timeStep):
        # Calculate which cell in a given column at the given time was active but not bursting.
        cellsActive = 0
        cellNumber = None
        for k in range(len(col.cells)):
            # Count the number of cells in the column that where active.
            if self.activeState(col, k, timeStep) is True:
                cellsActive += 1
                cellNumber = k
            if cellsActive > 1:
                break
        if cellsActive == 1 and cellNumber is not None:
            return cellNumber
        else:
            return None

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

    def getConnectedSynapses(self, column):
        # Create a list of the columns connected Synapses.
        column.connectedSynapses = np.array([], dtype=object)
        connSyn = []
        columnInd = column.pos_y * self.width + column.pos_x
        for i in range(len(self.colPotSynPerm[columnInd])):
            if self.colPotSynPerm[columnInd][i] > self.connectPermanence:
                # Update the synapses permanence from the colPotSynPerm matrix
                # This matrix hold all the synapse permanence values and is
                # updated by the learning calculator.
                column.potentialSynapses[i].permanence = self.colPotSynPerm[columnInd][i]
                # Add this synapse to the list pf connected synapses.
                connSyn.append(column.potentialSynapses[i])
        column.connectedSynapses = np.append(column.connectedSynapses, connSyn)

        return column.connectedSynapses

    def getColumnsOverlap(self, column):
        # Return the columns overlap value. This is stored in the
        # self.colOverlaps matrix.
        # columnIndex is the index number of the column if the
        # columns 2D array was flattened.
        columnIndex = column.pos_y * self.width + column.pos_x
        return self.colOverlaps[columnIndex]

    def getColumnsMinOverlap(self):
        # return the minoverlap value.
        # All columns have the same minoverlap value in a htm layer.
        return self.minOverlap

    def setupPotentialSynapses(self, inputWidth, inputHeight):
        # setup the locations of the potential synapses for every column.
        # Don't use this function to change the potential synapse list it
        # won't work as the theano class uses the intial parameters to
        # setup theano functions which workout the potential list.
        # Call the theano overlap class with the parameters
        # to obtain a list of x and y positions in the input that each
        # column can connect a potential synapse to.
        columnPotSynPositions = self.overlapCalc.getPotentialSynapsePos(inputWidth, inputHeight)

        # If the columns potential Width or height has changed then its
        # length of potential synapses will have changed. If not just change
        # each synpases parameters.
        cInd = 0
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                numPotSynapse = self.potentialHeight * self.potentialWidth
                assert numPotSynapse == len(columnPotSynPositions[0][0])
                c.potentialSynapses = np.array([])
                for i in range(numPotSynapse):
                    y = columnPotSynPositions[0][cInd][i]
                    x = columnPotSynPositions[1][cInd][i]
                    c.potentialSynapses = np.append(c.potentialSynapses,
                                                    [Synapse(x, y, -1, self.colSynPermanence)])
                cInd += 1

    def neighbours(self, c):
        # returns a list of the columns that are within the inhibitionRadius of c
        # Request from the inhibition calculator the neighbours of a particular
        # column. Return all the column that are neighbours
        columnIndex = c.pos_y * self.width + c.pos_x
        colIndicieList = self.inhibCalc.getColInhibitionList(columnIndex)
        print "colIndicieList = %s" % colIndicieList
        closeColumns = []
        allColumns = self.columns.flatten().tolist()
        for i in colIndicieList:
            # Convert the colIndicieList values from floats to ints.
            closeColumns.append(allColumns[int(i)])
        return np.array(closeColumns)

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

    def updateColActNotBurstTimes(self, columnIndex, timeStep):
        # update the vector holding the timestep when the columns
        # where active but not bursting last.
        # The input column c should be updated with the input timestep
        # unless it already has that timestep value. If this is the case
        # then no update should occur. Check the last position for the current time
        # and also check the temporary storing position which also may hold this time
        # if the column already tried to check this time.
        if (self.colActNotBurstTimes[columnIndex][0] != timeStep and
            self.colActNotBurstTimes[columnIndex][2] != timeStep):
            # move the vector back so the old time is kept.
            self.colActNotBurstTimes[columnIndex][1] = self.colActNotBurstTimes[columnIndex][0]
            self.colActNotBurstTimes[columnIndex][0] = timeStep
        else:
            # A cell in the column was already active so this column is bursting.
            # revert the latest timeback to the previous value.
            # Also store the current time in the temp position so the column knows this time
            # was already checked.
            self.colActNotBurstTimes[columnIndex][2] = timeStep
            self.colActNotBurstTimes[columnIndex][0] = self.colActNotBurstTimes[columnIndex][1]

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
        # The colActNotBurstTimes matrix is updated as well.
        columnIndex = c.pos_y * self.width + c.pos_x
        self.updateColActNotBurstTimes(columnIndex, timeStep)
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
                        synapseList.append(Synapse(m.pos_x, m.pos_y, j, self.cellSynPermanence))
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
        # if so return the number of synpases with the state
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
        # print " adaptSegments x,y,cell = %s,%s,%s positive
        # reinforcement = %r"%(c.pos_x,c.pos_y,i,positiveReinforcement)
        # Adds the new segments to the cell and inc or dec the segments synapses
        # If positive reinforcement is true then segments on the update list
        # get their permanence values increased all others
        # get their permanence decreased.
        # If positive reinforcement is false then decrement
        # the permanence value for the active synapses.
        for j in range(len(c.cells[i].segmentUpdateList)):
            # print "     segUpdateList = %s" % c.cells[i].segmentUpdateList[j]
            segIndex = c.cells[i].segmentUpdateList[j]['index']
            # print "     segIndex = %s"%segIndex
            # If the segment exists
            if segIndex > -1:
                # print "     adapted x,y,cell,segment=%s,%s,%s,%s"%(c.pos_x,c.pos_y,i,c.cells[i].segmentUpdateList[j]['index'])
                for s in c.cells[i].segmentUpdateList[j]['activeSynapses']:
                    # For each synapse in the segments activeSynapse list increment or
                    # decrement their permanence values.
                    # The synapses in the update segment
                    # structure are already in the segment. The
                    # new synapses are not yet however.
                    if positiveReinforcement is True:
                        s.permanence += self.permanenceInc
                        s.permanence = min(1.0, s.permanence)
                    else:
                        s.permanence -= self.permanenceDec
                        s.permanence = max(0.0, s.permanence)
                    # print "     x,y,cell,segment= %s,%s,%s,%s
                    # syn end x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,
                        # c.cells[i].segmentUpdateList[j]['index'],s.pos_x,
                        # s.pos_y,s.cell)
                    # print "     synapse permanence = %s"%(s.permanence)
                # Decrement the permanence of all synapses in the synapse list
                for s in c.cells[i].segments[segIndex].synapses:
                    s.permanence -= self.permanenceDec
                    s.permanence = max(0.0, s.permanence)
                    # print "     x,y,cell,segment= %s,%s,%s,%s syn end x,
                    # y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,j,
                    # s.pos_x,s.pos_y,s.cell)
                    # print "     synapse permanence = %s"%(s.permanence)
                # Add the new Synpases in the structure to the real segment
                # print c.cells[i].segmentUpdateList[j]['newSynapses']
                # print "oldActiveSyn = %s newSyn = %s"
                # %(len(c.cells[i].segments[segIndex].synapses),
                    # len(c.cells[i].segmentUpdateList[j]['newSynapses']))
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
        if type(newInput) == np.ndarray and len(newInput.shape) == 2:
            if newInput.shape != self.Input.shape:
                print "         New input size width,height = %s, %s" % (len(newInput[0]), len(newInput))
                print "         does not match old input size = %s,%s" % (len(self.Input[0]), len(self.Input))
            self.Input = newInput
        else:
            print "New Input is not a 2D numpy array!"

    def getPotentialOverlaps(self):
        # Get the potential overlap scores for each column.
        # Get them from the overlpas calculator
        return self.overlapCalc.getPotentialOverlaps()

    def getColActNotBurstVect(self):
        # Return the binary vector displaying if a column was active
        # but not bursting one timestep ago.
        # TODO
        pass

    def Overlap(self):
        """
        Phase one for the spatial pooler

        Calculate the overlap value each column has with it's
        connected input synapses.
        """

        # print "len(self.input) = %s len(self.input[0]) = %s " % (len(self.Input), len(self.Input[0]))
        # print "len(colPotSynPerm) = %s len(colPotSynPerm[0]) = %s" % (len(self.colPotSynPerm), len(self.colPotSynPerm[0]))
        # print "self.colPotSynPerm = \n%s" % self.colPotSynPerm
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        self.colOverlaps, self.colPotInputs = self.overlapCalc.calculateOverlap(self.colPotSynPerm, self.Input)
        # limit the overlap values so they are larger then minOverlap
        self.colOverlaps = self.overlapCalc.removeSmallOverlaps(self.colOverlaps)

    def inhibition(self, timeStep):
        '''
        Phase two for the spatial pooler

        Inhibit the weaker active columns.
        '''

        # The inhibitor calculator requires the column overlaps to be in
        # a grid with the same shape as the HTM layer.
        colOverlapsGrid = self.colOverlaps.reshape((self.height, self.width))
        # It also requires the potential overlaps to be in a grid form.
        # Get the potential overlaps and reshape them into a grid (matrix).
        potColOverlapsGrid = self.getPotentialOverlaps().reshape((self.height, self.width))

        self.colActive = self.inhibCalc.calculateWinningCols(colOverlapsGrid, potColOverlapsGrid)
        # print "self.colActive = \n%s" % self.colActive

        # Update the activeColumn list using the colActive vector.
        self.activeColumns = np.array([], dtype=object)
        allColumns = self.columns.flatten().tolist()
        indx = 0
        for c in allColumns:
            if self.colActive[indx] == 1:
                self.activeColumns = np.append(self.activeColumns, c)
                self.columnActiveAdd(c, timeStep)
            indx += 1

    def learning(self):
        '''
        Phase three for the spatial pooler

        Update the column synapses permanence.
        '''

        self.colPotSynPerm = self.learningCalc.updatePermanenceValues(self.colPotSynPerm,
                                                                      self.colPotInputs,
                                                                      self.colActive)

    def sequencePooler(self, timeStep):
        '''
        Calls the calculators that update the sequence pooler.
         1. update the active cells
         2. update the predictive cells
         3. perform learning on the selected distal synpases

        '''

        # 1. CALCULATE ACTIVE CELLS
        # Update the active cells and get the active Cells times from the calculator.
        self.activeCellsTime = self.activeCellsCalc.updateActiveCells(timeStep,
                                                                      self.colActive,
                                                                      self.predictiveCells,
                                                                      self.activeSeg,
                                                                      self.distalSynapses)

        # Get the update distal synapse tensors storing information on which
        # distal synapses should be updated from the active cells calculator class.
        (self.segIndUpdateActive,
         self.segActiveSynActive,
         self.segIndNewSynActive,
         self.segNewSynActive) = self.activeCellsCalc.getSegUpdates()

        # 2. CALCULATE PREDICTIVE CELLS
        self.predictCellsTime = self.predictCellsCalc.updatePredictiveState(timeStep,
                                                                            self.activeCellsTime,
                                                                            self.distalSynapses)
        # Get the update distal synapse tensors storing information on which
        # distal synapses should be updated for the predictive cells calculator.
        (self.segIndUpdatePredict,
         self.segActiveSynPredict) = self.predictCellsCalc.getSegUpdates()

    def updateActiveState(self, timeStep):
        # TODO
        # Remove this function it goes in a calculator class.
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

    def sequenceLearning(self, timeStep):
        # Third function called for the sequence pooler.
        # The update structures are implemented on the cells
        #print "\n       3rd SEQUENCE FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    # print "predictiveStateArray for x,y,i =
                    # %s,%s,%s is latest time = %s"%(c.pos_x,c.pos_y,i,
                        # c.predictiveStateArray[i,0])
                    if ((self.learnState(c, i, timeStep) is True) and
                        (self.learnState(c, i, timeStep-1) is False)):
                        # print "learn state for x,y,cell =
                        # %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c, i, True)
                    # Trying a different method to the CLA white pages
                    #if self.activeState(c,i,timeStep) ==
                    #False and self.predictiveState(c,i,timeStep-1) is True:
                    if ((self.predictiveState(c, i, timeStep-1) is True and
                        self.predictiveState(c, i, timeStep) is False and
                        self.activeState(c, i, timeStep) is False)):
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

    def temporal(self):
        '''
        Temporal Pooler keeps cells that are correclty predicitng a
        pattern active longer through the input sequence.

        It works by adjusting the permanences of both the proximal and distal syanpses.
        It performs learning on the column synapses and cell synapses for columsn and cells
        that where previously "active predictive" (they where predicting and then became active)
        and columsn that have just become active predictive.

        '''
        # TODO
        pass
        # We need just the latest column active but not bursting times.
        # latestColActNotBurstTimes = self.colActNotBurstTimes[:, 0]

        # self.colOverlaps, self.colStopTempAtTime = self.tempPoolCalc.calculateTemporalPool(latestColActNotBurstTimes,
        #                                                                                    self.timeStep,
        #                                                                                    self.colOverlaps,
        #                                                                                    self.colPotInputs,
        #                                                                                    self.colStopTempAtTime)


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
        # Enable space in the first layers input for feedback from higher levels.
        self.enableHigherLevFb = params['enableHigherLevFb']
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
            # print "predCommGrid = %s" % predCommGrid
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

    def commandSpaceOutput(self, layer):
        # Return the output from the command space
        # This is the top half of the output from the selected layer
        layerHeight = self.layerArray[layer].height
        wholeOutput = self.layerArray[layer].output
        halfLayerHeight = int(layerHeight/2)

        commSpaceOutput = wholeOutput[0:halfLayerHeight, :]
        return commSpaceOutput

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

    def spatialTemporal(self):
        i = 0
        for layer in self.layerArray:
            # print "     Layer = %s" % i
            i += 1
            layer.timeStep = layer.timeStep+1
            # Update the current layers input with the new input
            # This updates the spatial pooler
            layer.Overlap()
            layer.inhibition(layer.timeStep)
            layer.learning()
            # This updates the sequence pooler
            layer.sequencePooler(layer.timeStep)
            # layer.updateActiveState(layer.timeStep)
            # layer.updatePredictiveState(layer.timeStep)
            # layer.sequenceLearning(layer.timeStep)
            # TODO
            # This updates the temporal pooler
            #layer.temporal()


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

        ### Setup the inputs and outputs between levels
        # Each regions input needs to make room for the command
        # feedback from the higher level.
        commandFeedback = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                                    for j in range(int(self.height/2))])
        # The lowest region receives the new input.
        # If the region has enablehigherLevFb parameter enabled add extra space to the input.
        if bottomRegionsParams['enableHigherLevFb'] == 1:
            newInput = SDRFunct.joinInputArrays(commandFeedback, input)
        else:
            newInput = input
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
            # If the region has higherLevFb param enabled add extra space to the input.
            if regionsParam['enableHigherLevFb'] == 1:
                newInput = SDRFunct.joinInputArrays(commandFeedback, lowerOutput)
            else:
                newInput = lowerOutput

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
        # Save the synapses for the command area so they can be reloaded.
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
        if self.regionArray[0].enableHigherLevFb == 1:
            if self.numLevels > 1:
                commFeedbackLev1 = self.levelOutput(1)
            else:
                # This is the highest level but the enable higher level feedback
                # command is enabled. In this case we will just use the current levels command.
                commFeedbackLev1 = self.levelOutput(0)
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
            if self.regionArray[i].enableHigherLevFb == 1:
                # Check to make sure this isn't the highest level
                if higherLevel < self.numLevels:
                    # Get the feedback command from the higher level
                    commFeedbackLevN = self.levelOutput(higherLevel)
                else:
                    # This is the highest level but the enable higher level feedback
                    # command is enabled. In this case we will just use the current levels command.
                    commFeedbackLevN = self.levelOutput(i)

            # Update the newInput for the current level in the HTM
            newInput = SDRFunct.joinInputArrays(commFeedbackLevN, lowerLevelOutput)
            self.regionArray[i].updateRegionInput(newInput)

    def levelOutput(self, level):
        # Return the output from the desired level.
        # The output will be from the highest layer in the level.
        highestLayer = self.regionArray[level].numLayers-1
        #return self.regionArray[level].layerOutput(highestLayer)
        return self.regionArray[level].commandSpaceOutput(highestLayer)
        #return self.regionArray[level].regionOutput()

    def updateAllThalamus(self):
        # Update all the thalaums classes in each region
        for i in range(self.numLevels):
            # TODO
            # HAck to make higher levels thalamus choose commands slower
            if self.regionArray[i].layerArray[0].timeStep % (2*i+1) == 0:
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




