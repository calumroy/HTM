# Title: HTM layer
# Description: HTM layer class and other sub classes
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
from HTM_calc import theano_learning as spatLearning
#from HTM_calc import np_learning as spatLearning

from HTM_calc import np_activeCells as activeCells
from HTM_calc import theano_predictCells as predictCells
from HTM_calc import np_sequenceLearning as seqLearn


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
        # Enable space in the input for feedback from other levels.
        from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        self.enableFeedback = params['enableFeedback']
        # An index indicating the level that feedback will come from.
        self.feedbackLevelInd = params['feedbackLevelInd']
        # An index indicating the layer in a particular level that feedback will come from.
        self.feedbackLayerInd = params['feedbackLayerInd']

        # How many columns within the inhibition radius are active
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
        # Setup a vector where each element represents if a column is active 1 or not 0
        self.colActive = np.zeros(self.numColumns)
        # Setup a vector where each element represents if a column was active one timeStep ago
        # 1 represens active, 0 not active one timeStep ago
        self.prevColActive = np.zeros(self.numColumns)
        # The timeSteps when columns where last bursting. This is a 2D tensor.
        # The 1st dimension stores for each column the last 2 timesteps when it was last bursting.
        self.burstColsTime = np.array([[-1, -1] for y in range(self.numColumns)])

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
            for x in range(len(self.columns[0])):
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
        #print "self.colActive = \n%s" % self.colActive

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
        # Get the timeSteps when each column was bursting last.
        self.burstColsTime = self.activeCellsCalc.getBurstCol()

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

        self.colPotSynPerm = self.tempPoolCalc.updateProximalTempPool(self.colPotInputs,
                                                                      self.colActive,
                                                                      self.colPotSynPerm,
                                                                      self.timeStep,
                                                                      self.activeCellsTime,
                                                                      self.burstColsTime
                                                                      )

        # This updates distal synapses causing some cells to predict more often.
        self.distalSynapses = self.tempPoolCalc.updateDistalTempPool(self.timeStep,
                                                                     self.learnCellsList,
                                                                     self.learnCellsTime,
                                                                     self.predictCellsTime,
                                                                     self.activeCellsTime,
                                                                     self.activeSegsTime,
                                                                     self.distalSynapses)


