# Title: HTM
# Description: HTM network
# Author: Calum Meiklejohn
# Development phase: alpha

import cProfile
import numpy as np
import random
import math
import copy
from utilities import sdrFunctions as SDRFunct
from reinforcement_learning import Thalamus

from HTM_calc import np_temporal as temporal
from HTM_calc import theano_overlap as overlap
# from HTM_calc import theano_inhibition as inhibition
from HTM_calc import np_inhibition as inhibition
# from HTM_calc import theano_learning as learning
from HTM_calc import np_learning as spatLearning

from HTM_calc import np_activeCells as activeCells
from HTM_calc import np_predictCells as predictCells
from HTM_calc import np_sequenceLearning as seqLearn


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
        # Stores the last time step that this segment was predicting activity
        self.sequenceSegment = 0
        # Stores the synapses that have been created and
        # have a larger permenence than 0.0
        self.synapses = []


class Cell:
    def __init__(self):
        self.score = 0     # The current score for the cell.
        self.segments = []


class Column:
    def __init__(self, length, pos_x, pos_y):
        self.cells = [Cell() for i in range(length)]
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.overlap = 0.0  # As defined by the numenta white paper
        # An array storing the synapses with a permanence greater then the connectPermanence.
        self.connectedSynapses = np.array([], dtype=object)
        # The possible feed forward Synapse connections for the column
        self.potentialSynapses = np.array([], dtype=object)


class HTMLayer:
    def __init__(self, input, columnArrayWidth, columnArrayHeight, cellsPerColumn, params):
        # The columns are in a 2 dimensional array columnArrayWidth by columnArrayHeight.
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
        self.timeStep = 0
        # The output is a 2D grid representing the cells states.
        # It is larger then the input by a factor of the number of cells per column
        self.output = np.zeros((self.height, self.width * self.cellsPerColumn))
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
        self.potentialWidth = params['potentialWidth']
        self.potentialHeight = params['potentialHeight']
        self.minOverlap = params['minOverlap']
        self.wrapInput = params['wrapInput']
        self.spatialPermanenceInc = params['spatialPermanenceInc']
        self.spatialPermanenceDec = params['spatialPermanenceDec']
        self.tempPermanenceInc = params['spatialPermanenceInc']
        self.tempPermanenceDec = params['spatialPermanenceDec']
        self.tempSpatialPermanenceInc = params['tempSpatialPermanenceInc']
        self.tempSeqPermanenceInc = params['tempSeqPermanenceInc']
        self.tempDelayLength = params['tempDelayLength']
        # Already active columns have their spatial synapses decremented by a different value in the spatial pooler.
        self.activeColPermanenceDec = params['activeColPermanenceDec']
        self.permanenceInc = params['permanenceInc']
        self.permanenceDec = params['permanenceDec']
        self.inputHeight = len(self.Input)
        self.inputWidth = len(self.Input[0])
        # Check that the potential width and height are of odd shape if the wrapInput parameter is true.
        self.setWrapPotPoolShape()

        self.numPotSyn = self.potentialWidth * self.potentialHeight
        self.numColumns = self.height * self.width
        # Setup a matrix where each row represents a list of a columns
        # potential synapse permanence values
        self.colPotSynPerm = np.array([[self.colSynPermanence for i in range(self.numPotSyn)]
                                      for j in range(self.numColumns)])
        # Setup a matrix where each position represents a columns overlap.
        # This only checks synapses that are connected.
        self.colOverlaps = np.empty([self.height, self.width])
        # Setup a matrix where each position represents a columns potential overlap.
        # All potential synapses are checked not just the connected ones.
        self.potColOverlapsGrid = np.empty([self.height, self.width])
        # Setup a matrix where each row represents a columns input values from its potential synapses.
        self.colPotInputs = np.empty([self.numColumns, self.numPotSyn])
        # Store the previous timeSteps potential inputs.
        self.prevColPotInputs = np.empty([self.numColumns, self.numPotSyn])
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
        # Setup a vector where each element represents if a column is active 1 or not 0
        self.colActive = np.zeros(self.numColumns)
        # Setup a vector where each element represents if a column was active one timeStep ago
        # 1 represens active, 0 not active one timeStep ago
        self.prevColActive = np.zeros(self.numColumns)

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

        # The timeSteps when cells where in the learn state last. This is a 3D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timestep when this cell was in the learn state last.
        self.learnCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)]
                                       for y in range(self.numColumns)])

        # A variable length list storing the column Index and cell index of the learning cells
        # for the current timeStep.
        self.learnCellsList = []


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
        # activeSegsTime is a 3D tensor. The first dimension is the columns, the second the
        # cells and the 3rd is the segment in the cells. For each segment a timeStep is stored
        # indicating when the segment was last in an active state. This means it was
        # predicting that the cell would become active in the next timeStep.
        # This is what the CLA paper calls a "SEQUENCE SEGMENT".
        self.activeSegsTime = np.zeros((self.numColumns, self.cellsPerColumn, self.maxNumSegments))

        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # This tensor stores segment update info from the active Cells calculator.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdateActive = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        # This tensor stores segment update info from the active Cells calculator.
        self.segActiveSynActive = np.array([[[-1 for z in range(self.newSynapseCount)]
                                            for x in range(self.cellsPerColumn)]
                                            for y in range(self.numColumns)])
        # A 2D tensor "segIndNewSynActive" for each cell holds [segIndex] indicating which segment new
        # synapses should be created for. If the index is -1 don't create any new synapses.
        # This tensor stores segment update info from the active Cells calculator.
        self.segIndNewSynActive = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 4D tensor "segNewSyn" for each cell holds a synapse list [newSynapseList] of new synapses
        # that could possibly be created. Each position corresponds to a synapses in the segment
        # with the index stored in the segNewSynActive tensor.
        # Each place in the newSynapseList holds [columnIndex, cellIndex, permanence]
        # If permanence is -1 then this means don't create a new synapse for that synapse.
        # This tensor stores segment update info from the active Cells calculator.
        self.segNewSynActive = np.array([[[[-1, -1, 0.0] for z in range(self.newSynapseCount)]
                                        for x in range(self.cellsPerColumn)]
                                        for y in range(self.numColumns)])
        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # This tensor stores segment update info from the predict Cells calculator.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdatePredict = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        # This tensor stores segment update info from the predictive Cells calculator.
        self.segActiveSynPredict = np.array([[[-1 for z in range(self.newSynapseCount)]
                                            for x in range(self.cellsPerColumn)]
                                            for y in range(self.numColumns)])

        # Create the array storing the columns
        self.columns = np.array([[]], dtype=object)
        # Setup the columns array.
        self.setupColumns()

        # Setup the theano classes used for calculating
        # spatial, temporal and sequence pooling.
        self.setupCalculators()

        # Initialise the columns potential synapses.
        # Work out the potential feedforward connections each column could make to the input.
        self.setupPotentialSynapses(self.inputWidth, self.inputHeight)

    def setWrapPotPoolShape(self):
        # If the wrapInput parameter is true then set the potential pool shapes to odd.
        # This is because the wrap function won't work with even shapes since the kernal can't be cenetered.
        # This is actually a theano restriction (fix this in the future).
        if self.wrapInput == True:
            if self.potentialHeight % 2 != 1:
                self.potentialHeight = self.potentialHeight + 1
                if self.potentialHeight > self.inputHeight:
                    self.potentialHeight = self.inputHeight
                if self.potentialHeight % 2 != 1:
                    self.potentialHeight = self.inputHeight - 1
                print "WARNING: The columns potential height was changed to %s" %self.potentialHeight
                print "     The overlap calculators wrapping function requires odd pooling shapes smaller then the input."
            if self.potentialWidth % 2 != 1:
                self.potentialWidth = self.potentialWidth + 1
                if self.potentialWidth > self.inputWidth:
                    self.potentialWidth = self.inputWidth
                if self.potentialWidth % 2 != 1:
                    self.potentialWidth = self.inputWidth - 1
                print "WARNING: The columns potential width was changed to %s" %self.potentialWidth
                print "     The overlap calculators wrapping function requires odd pooling shapes smaller then the input."

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
                                                     self.minOverlap,
                                                     self.wrapInput)

        self.inhibCalc = inhibition.inhibitionCalculator(self.width, self.height,
                                                         self.inhibitionWidth,
                                                         self.inhibitionHeight,
                                                         self.desiredLocalActivity,
                                                         self.minOverlap,
                                                         self.centerPotSynapses)

        self.spatLearningCalc = spatLearning.LearningCalculator(self.numColumns,
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
                                                                 self.cellSynPermanence,
                                                                 self.connectPermanence)

        self.predictCellsCalc = predictCells.predictCellsCalculator(self.numColumns,
                                                                    self.cellsPerColumn,
                                                                    self.maxNumSegments,
                                                                    self.newSynapseCount,
                                                                    self.connectPermanence,
                                                                    self.activationThreshold)

        self.seqLearnCalc = seqLearn.seqLearningCalculator(self.numColumns,
                                                           self.cellsPerColumn,
                                                           self.maxNumSegments,
                                                           self.newSynapseCount,
                                                           self.connectPermanence,
                                                           self.permanenceInc,
                                                           self.permanenceDec)

        self.tempPoolCalc = temporal.TemporalPoolCalculator(self.cellsPerColumn,
                                                            self.numColumns,
                                                            self.numPotSyn,
                                                            self.tempSpatialPermanenceInc,
                                                            self.tempSeqPermanenceInc,
                                                            self.minThreshold,
                                                            self.cellSynPermanence,
                                                            self.connectPermanence,
                                                            self.tempDelayLength)

    def setupColumns(self):
        # Get just the parameters for the columns
        # Note: The parameters can come in a list of dictionaries,
        # one for each column or a shorter list specifying only some columns.
        # If only one or a few columns have parameters specified then all the
        # rest of the columns get the same last parameters specified.
        self.columns = np.array([[Column(self.cellsPerColumn, i, j)
                                for i in range(self.width)] for
                                j in range(self.height)], dtype=object)

    def neighbours(self, c):
        # returns a list of the columns that are within the inhibitionRadius of c
        # Request from the inhibition calculator the neighbours of a particular
        # column. Return all the column that are neighbours
        columnIndex = c.pos_y * self.width + c.pos_x
        colIndicieList = self.inhibCalc.getColInhibitionList(columnIndex)
        #print "Columns Neighbours list = %s" % colIndicieList
        closeColumns = []
        allColumns = self.columns.flatten().tolist()
        for i in colIndicieList:
            # Convert the colIndicieList values from floats to ints.
            closeColumns.append(allColumns[int(i)])
        return np.array(closeColumns)

    def getActiveColumnsGrid(self):
        # Return a matrix with the same dimensions as the 2d htm layer,
        # where each element represents a column. 1 means the column is active 0 it is not.
        activeColumns = self.colActive.reshape((self.width, self.height))
        return activeColumns

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

    # def activeNotBurstCellGrid(self):
    #     # Return a grid representing the cells in the columns which are active but
    #     # not bursting. Cells in a column are placed in adjacent grid cells right of each other.
    #     # Eg. A HTM layer with 10 rows, 5 columns and 3 cells per column would produce an
    #     # activeCellGrid of 10*3 = 15 columns and 10 rows.
    #     output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
    #                       for j in range(self.height)])
    #     for c in self.activeColumns:
    #         x = c.pos_x
    #         y = c.pos_y
    #         cellsActive = 0
    #         cellNumber = None
    #         for k in range(len(c.cells)):
    #             # Count the number of cells in the column that where active.
    #             if c.activeStateArray[k][0] == self.timeStep:
    #                 cellsActive += 1
    #                 cellNumber = k
    #             if cellsActive > 1:
    #                 break
    #         if cellsActive == 1 and cellNumber is not None:
    #             output[y][x*self.cellsPerColumn+cellNumber] = 1
    #     #print "output = ", output
    #     return output

    def checkCellActive(self, pos_x, pos_y, cellIndex, timeStep):
        # Find if the given cell is active at the given timeStep
        # We check the self.activeCellsTime tensor which stores the last 2
        # times each cell was active.
        colIndex = pos_y * self.width + pos_x
        if self.activeCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if self.activeCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def checkCellPredict(self, pos_x, pos_y, cellIndex, timeStep):
        # Find if the given cell is in the predictive state at the given timeStep
        # We check the self.predictCellsTime tensor which stores the last 2
        # times each cell was active.
        colIndex = pos_y * self.width + pos_x
        if self.predictCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if self.predictCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def checkCellLearn(self, pos_x, pos_y, cellIndex, timeStep):
        # Find if the given cell is in the learn state at the given timeStep
        # We check the self.learnCellsTime tensor which stores the last 2
        # times each cell was active.
        colIndex = pos_y * self.width + pos_x
        if self.learnCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if self.learnCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def getCellsUpdateSynStruct(self, pos_x, pos_y, cellInd):
        # The update synapse structures for the given cell
        colInd = pos_y * self.width + pos_x
        print " cells self.segIndUpdateActive = %s " % self.segIndUpdateActive[colInd][cellInd]
        print " cells self.segActiveSynActive = %s " % self.segActiveSynActive[colInd][cellInd]
        print " cells self.segIndNewSynActive = %s " % self.segIndNewSynActive[colInd][cellInd]
        print " cells self.segNewSynActive = %s " % self.segNewSynActive[colInd][cellInd]
        print " cells self.segIndUpdatePredict = %s " % self.segIndUpdatePredict[colInd][cellInd]
        print " cells self.segActiveSynPredict = %s " % self.segActiveSynPredict[colInd][cellInd]

    def getCellsScore(self, pos_x, pos_y, cellInd):
        # Get the score of the selected cell from the activeCells calculator
        colInd = pos_y * self.width + pos_x
        cellsScore = self.activeCellsCalc.getCellsScore(colInd, cellInd)
        return cellsScore

    def getNumSegments(self, pos_x, pos_y, cellInd):
        # Get the number of segments for a cell in the column at position
        # pos_x, pos_y with index cellInd.
        colInd = pos_y * self.width + pos_x

        segList = self.distalSynapses[colInd][cellInd]

        # Now we need to check if the segment has been created yet or if
        # the segment is just a placeholder and has never been activated.
        # Check that self.distalSynapse tensor for segments that contain any
        # synpases with permanence values larger then a zero permanence value.
        numSegments = 0
        # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        for segInd in range(len(segList)):
            synList = self.distalSynapses[colInd][cellInd][segInd]
            for synInd in range(len(synList)):
                synPermanence = synList[synInd][2]
                if synPermanence > 0:
                    # This segment contains initialised synapses.
                    numSegments += 1
                    break
        return numSegments

    def getConnectedCellsSegSyns(self, column, cellInd, segInd):
        # Return an array of synpases objects that represent the distal synapses
        # in a particular segment.
        # The column is an object the cellInd, and segInd are indicies.
        # get the columns index in the self.distalSynapses tensor (5d tensor).
        # Each synapse stores (column number, cell number, permanence)
        colInd = column.pos_y * self.width + column.pos_x

        synList = self.distalSynapses[colInd][cellInd][segInd]
        # print "synList = \n%s" % (synList)

        # Now we need to check if the synapse is connected.
        connectedSynList = []
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        for syn in synList:
            synPermanence = syn[2]
            if synPermanence > self.connectPermanence:
                # Create a synapse object to represent this synapse.
                # Convert the column index to a column x, y position
                endColInd = syn[0]
                endCellInd = syn[1]
                col_pos_y = math.floor(int(endColInd) / int(self.width))
                col_pos_x = endColInd - col_pos_y * self.width

                newSynObj = Synapse(col_pos_x, col_pos_y, endCellInd, synPermanence)
                connectedSynList.append(newSynObj)

        return connectedSynList

    def predictiveCellGrid(self):
        # Return a grid representing the cells in the columns which are predicting.
        output = np.array([[0 for i in range(self.width*self.cellsPerColumn)]
                          for j in range(self.height)])
        for y in range(len(self.columns)):
            for x in range(len(self.columns[i])):
                c = self.columns[y][x]
                for k in range(len(c.cells)):
                    # Set the cells in the column that are predicting now
                    if self.checkCellPredict(x, y, k, self.timeStep):
                        output[y][x*self.cellsPerColumn+k] = 1
        # print "output = ", output
        return output

    def setupPotentialSynapses(self, inputWidth, inputHeight):
        # setup the locations of the potential synapses for every column.
        # Don't use this function to change the potential synapse list it
        # won't work as the theano class uses the intial parameters to
        # setup theano functions which workout the potential list.
        # Call the theano overlap class with the parameters
        # to obtain a list of x and y positions in the input that each
        # column can connect a potential synapse to.
        columnPotSynPositions = self.overlapCalc.getPotentialSynapsePos(inputWidth, inputHeight)
        numPotSynapse = self.potentialHeight * self.potentialWidth

        # Just create a potential proximal synpase object for each of the potential synapses.
        cInd = 0
        for k in range(len(self.columns)):
            for c in self.columns[k]:

                assert numPotSynapse == len(columnPotSynPositions[0][0])
                c.potentialSynapses = np.array([])
                for i in range(numPotSynapse):
                    y = columnPotSynPositions[0][cInd][i]
                    x = columnPotSynPositions[1][cInd][i]
                    c.potentialSynapses = np.append(c.potentialSynapses,
                                                    [Synapse(x, y, -1, self.colSynPermanence)])
                cInd += 1

    def getPotColSynapses(self, column):
        # Get the list of the columns potential Synapses.
        columnInd = column.pos_y * self.width + column.pos_x
        for i in range(len(self.colPotSynPerm[columnInd])):
            # Update the synapses permanence from the colPotSynPerm matrix
            # This matrix hold all the synapse permanence values and is
            # updated by the learning calculator.
            column.potentialSynapses[i].permanence = self.colPotSynPerm[columnInd][i]
        return column.potentialSynapses

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
                # Add this synapse to the list of connected synapses.
                connSyn.append(column.potentialSynapses[i])
        column.connectedSynapses = np.append(column.connectedSynapses, connSyn)

        return column.connectedSynapses

    def getPotentialOverlaps(self, column=None):
        # Get the potential overlap scores for all columns if the column object
        # input to this function is None.
        # If a valid column object is requested return the potential overlap
        # score just for that column.
        # Get them from the overlaps calculator
        if column is None:
            return self.overlapCalc.getPotentialOverlaps()
        else:
            columnIndex = column.pos_y * self.width + column.pos_x
            return self.overlapCalc.getPotentialOverlaps()[columnIndex]

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

    def columnActiveStateNow(self, pos_x, pos_y):
        # look at the colActive array to see if a column is active at the moment.
        columnIndex = pos_y * self.width + pos_x
        if self.colActive[columnIndex] == 1:
            return True
        return False

    def findLearnCell(self, c, timeStep):
        # Return the cell index that was in the learn state in the column c at
        # the timeStep provided.
        # If no cell was in the learn state then return None
        learnCell = None
        for i in range(self.cellsPerColumn):
            if self.learnState(c, i, timeStep) is True:
                learnCell = i
        return learnCell

    def updateInput(self, newInput):
        # Update the input
        # Check to see if this input is the same size as the last and is a 2d numpy array
        if type(newInput) == np.ndarray and len(newInput.shape) == 2:
            if newInput.shape != self.Input.shape:
                print "         New input size width,height = %s, %s" % (len(newInput[0]), len(newInput))
                print "         does not match old input size = %s,%s" % (len(self.Input[0]), len(self.Input))
            self.Input = newInput
        else:
            print "New Input is not a 2D numpy array!"

    def updateOutput(self):
        # Update the output array.
        # The output array is the output from all the cells. The cells form a new 2d input grid
        # this way temporal information is not lost between layers and levels.
        # Initialise all outputs as zero first then set the cells as 1.
        self.output = np.zeros((self.height, self.width * self.cellsPerColumn))
        # Use the active cells from the activeCells calculator.
        # Set the corresponding bit in the output to true if the cell is currently active.
        # Active cells list returns a list of column indicies and cell indices which are active
        # [colInd, cellInd]
        activeCellsList = self.activeCellsCalc.getActiveCellsList()

        for activeCellPos in activeCellsList:
            colInd = activeCellPos[0]
            cellInd = activeCellPos[1]

            col_pos_y = int(math.floor(int(colInd) / int(self.width)))
            col_pos_x = colInd - col_pos_y * self.width
            # Output is a 2d grid where each location represents a cell.
            # Set the cell respective element in the output to 1 if that cell is active.
            self.output[col_pos_y][int(col_pos_x*self.cellsPerColumn+cellInd)] = 1

    def getLearnCellsOutput(self):
        # Update an output array representing all the cells in the learning state.
        # The output array is the output from all the cells. The cells form a new 2d input grid
        # Initialise all outputs as zero first then set the cells that are in the learning state as 1.
        learnStateOutput = np.zeros((self.height, self.width * self.cellsPerColumn))
        # Use the learning cells from the activeCells calculator.
        # Set the corresponding bit in the output to true if the cell is currently learning.
        # Learning cells list is a list of column indicies and cell indices which are learning.
        # [colInd, cellInd]
        
        for learnCellPos in self.learnCellsList:
            colInd = learnCellPos[0]
            cellInd = learnCellPos[1]

            col_pos_y = int(math.floor(int(colInd) / int(self.width)))
            col_pos_x = colInd - col_pos_y * self.width
            # Output is a 2d grid where each location represents a cell.
            # Set the cell respective element in the output to 1 if that cell is learning.
            learnStateOutput[col_pos_y][int(col_pos_x*self.cellsPerColumn+cellInd)] = 1
        return learnStateOutput

    def Overlap(self):
        """
        Phase one for the spatial pooler

        Calculate the overlap value each column has with it's
        connected input synapses.
        """

        # print "len(self.input) = %s len(self.input[0]) = %s " % (len(self.Input), len(self.Input[0]))
        # print "len(colPotSynPerm) = %s len(colPotSynPerm[0]) = %s" % (len(self.colPotSynPerm), len(self.colPotSynPerm[0]))
        # print "self.colPotSynPerm = \n%s" % self.colPotSynPerm
        # Save the previous potential Inputs
        self.prevColPotInputs = self.colPotInputs
        # Update the overlap and potential input values for each column.
        self.colOverlaps, self.colPotInputs = self.overlapCalc.calculateOverlap(self.colPotSynPerm, self.Input)
        # limit the overlap values so they are larger then minOverlap
        self.colOverlaps = self.overlapCalc.removeSmallOverlaps(self.colOverlaps)

        # Also get the potential overlaps in a grid form.
        # Get the potential overlaps and reshape them into a grid (matrix).
        self.potColOverlapsGrid = self.getPotentialOverlaps().reshape((self.height, self.width))

    def inhibition(self, timeStep):
        '''
        Phase two for the spatial pooler

        Inhibit the weaker active columns.
        '''

        # The inhibitor calculator requires the column overlaps to be in
        # a grid with the same shape as the HTM layer.
        colOverlapsGrid = self.colOverlaps.reshape((self.height, self.width))

        # Store the current activecol as the previous active columns and update the colActive
        self.prevColActive = self.colActive
        self.colActive = self.inhibCalc.calculateWinningCols(colOverlapsGrid, self.potColOverlapsGrid)
        # print "self.colActive = \n%s" % self.colActive

    def spatialLearning(self):
        '''
        Phase three for the spatial pooler

        Update the column synapses permanence.
        '''

        self.colPotSynPerm = self.spatLearningCalc.updatePermanenceValues(self.colPotSynPerm,
                                                                          self.colPotInputs,
                                                                          self.colActive)

    def sequencePooler(self, timeStep):
        '''
        Calls the calculators that update the sequence pooler.
         1. update the active cells
         2. update the predictive cells
         3. perform learning on the selected distal synpases

        '''
        self.calcActiveCells(timeStep)
        self.calcPredictCells(timeStep)
        self.sequenceLearning(timeStep)

    def calcActiveCells(self, timeStep):
        # 1. CALCULATE ACTIVE CELLS
        # Update the active cells and get the active Cells times and learn state times
        # from the calculator.
        (self.activeCellsTime,
         self.learnCellsTime) = self.activeCellsCalc.updateActiveCells(timeStep,
                                                                       self.colActive,
                                                                       self.predictCellsTime,
                                                                       self.activeSegsTime,
                                                                       self.distalSynapses)

        # Get the update distal synapse tensors storing information on which
        # distal synapses should be updated from the active cells calculator class.
        (self.segIndUpdateActive,
         self.segActiveSynActive,
         self.segIndNewSynActive,
         self.segNewSynActive) = self.activeCellsCalc.getSegUpdates()
        # Get the cells that are in the learning state as a list.
        self.learnCellsList = self.activeCellsCalc.getCurrentLearnCellsList()

    def calcPredictCells(self, timeStep):
        # 2. CALCULATE PREDICTIVE CELLS
        self.predictCellsTime = self.predictCellsCalc.updatePredictiveState(timeStep,
                                                                            self.activeCellsTime,
                                                                            self.distalSynapses)
        # Get the timeSteps for which each segment was last active
        self.activeSegsTime = self.predictCellsCalc.getActiveSegTimes()
        # Get the update distal synapse tensors storing information on which
        # distal synapses should be updated for the predictive cells calculator.
        (self.segIndUpdatePredict,
         self.segActiveSynPredict) = self.predictCellsCalc.getSegUpdates()

    def sequenceLearning(self, timeStep):
        # 3. CALCULATE THE UPDATED DISTAL SYNAPSES FOR SEQUENCE LEARNING
        self.distalSynapses = self.seqLearnCalc.sequenceLearning(timeStep,
                                                                 self.activeCellsTime,
                                                                 self.learnCellsTime,
                                                                 self.predictCellsTime,
                                                                 self.distalSynapses,
                                                                 self.segIndUpdateActive,
                                                                 self.segActiveSynActive,
                                                                 self.segIndNewSynActive,
                                                                 self.segNewSynActive,
                                                                 self.segIndUpdatePredict,
                                                                 self.segActiveSynPredict)

    def temporalPooler(self, timeStep):
        '''
        Temporal Pooler keeps cells that are correctly predicitng a
        pattern active longer through the input sequence.

        It works by adjusting the permanences of both the proximal and distal syanpses.
        It performs learning on the column synapses and cell synapses for columns and cells
        that where previously "active predictive" (they where predicting and then became active)
        and columns that have just become active predictive.

        '''

        #latestColActNotBurstTimes = self.colActNotBurstTimes[:, 0]

        self.colPotSynPerm = self.tempPoolCalc.updateProximalTempPool(self.colPotInputs,
                                                                      self.colActive,
                                                                      self.colPotSynPerm,
                                                                      self.timeStep,
                                                                      self.activeCellsTime
                                                                      )

        # This updates distal synapses causing some cells to predict more often.
        self.distalSynapses = self.tempPoolCalc.updateDistalTempPool(self.timeStep,
                                                                     self.learnCellsList,
                                                                     self.learnCellsTime,
                                                                     self.predictCellsTime,
                                                                     self.activeCellsTime,
                                                                     self.activeSegsTime,
                                                                     self.distalSynapses)


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
        # Return the output from the entire region.
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
            layer.spatialLearning()
            # This updates the sequence pooler
            layer.sequencePooler(layer.timeStep)
            # This updates the temporal pooler
            layer.temporalPooler(layer.timeStep)
            # Update the output grid for the layer.
            layer.updateOutput()


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




