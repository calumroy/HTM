
import random
import numpy as np

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

'''
A class to calculate the temporal pooling for a HTM layer.


'''


class TemporalPoolCalculator():
    def __init__(self, cellsPerColumn, numColumns, numPotSyn,
                 spatialPermanenceInc, spatialPermanenceDec,
                 seqPermanenceInc, seqPermanenceDec,
                 minNumSynThreshold, newSynPermanence,
                 connectPermanence, delayLength):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        self.numPotSynapses = numPotSyn
        # The value by which the spatial poolers synapses (proximal synapses)
        # permanence values change by.
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec
        # The value by which the sequence poolers synapses (distal synapses)
        # permanence values change by.
        self.seqPermanenceInc = seqPermanenceInc
        self.seqPermanenceDec = seqPermanenceDec
        # More then this many synapses in a segment must be active for the segment
        # to be considered a potnetial best matching segment and it getting it's synapse
        # permanence values increased.
        self.minNumSynThreshold = minNumSynThreshold
        # The starting permanence of new cell synapses. This is used to create new synapses.
        self.newSynPermanence = newSynPermanence
        # The minimum required permanence value required by a synapse for it to be connected.
        self.connectPermanence = connectPermanence

        # delayLength, a parameter for updating the average persistance count.
        # It determines how quickly the average persistance count changes.
        self.delayLength = delayLength

        # Store the previous colPotInputs.
        # This is so a potential synapse can work out if it's end
        # has changed state. If so then we update the synapses permanence.
        # Initialize with a negative value. Normally this matrix holds 0 or 1 only.
        self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevColActive = np.array([-1 for i in range(self.numColumns)])

        # Save the calculated vector describing if each column
        # was active but not bursting one timestep ago.
        self.colActNotBurstVect = None

        # A tracking number indicating how many times a cell has succesfully been
        # active predicted in a row.
        self.cellsTrackingNum = np.array([[0 for x in range(self.cellsPerColumn)]
                                          for y in range(self.numColumns)])
        # A number indicating how many times a cell usually stays active for.
        self.cellsAvgPersist = np.array([[-1 for x in range(self.cellsPerColumn)]
                                         for y in range(self.numColumns)])
        # A number indicating how many timeSteps a cell will stay predicting
        # into the future without any active segments.
        self.cellsPersistance = np.array([[-1 for x in range(self.cellsPerColumn)]
                                         for y in range(self.numColumns)])

    def checkCellActive(self, colIndex, cellIndex, timeStep, activeCellsTime):
        # Check if the given cell was active at the timestep given.
        # We need to check the activeCellsTime tensor which holds multiple
        # previous timeSteps when each cell was last active.
        # import ipdb; ipdb.set_trace()
        colIndex = int(colIndex)
        cellIndex = int(cellIndex)
        if activeCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if activeCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def setPredictCell(self, colIndex, cellIndex, timeStep, predictCellsTime):
        # Set the given cell at colIndex, cellIndex into a predictive state for the
        # given timeStep.
        # We need to check the predictCellsTime tensor which holds multiple
        # previous timeSteps and set the oldest one to the given timeStep.
        if predictCellsTime[colIndex][cellIndex][0] <= predictCellsTime[colIndex][cellIndex][1]:
            predictCellsTime[colIndex][cellIndex][0] = timeStep
        else:
            predictCellsTime[colIndex][cellIndex][1] = timeStep

    def checkCellPredict(self, colIndex, cellIndex, timeStep, predictCellsTime):
        # Check if the given cell was predicting at the timestep given.
        # We need to check the predictCellsTime tensor which holds multiple
        # previous timeSteps when each cell was last predicting.
        # import ipdb; ipdb.set_trace()
        colIndex = int(colIndex)
        cellIndex = int(cellIndex)
        if predictCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if predictCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def checkCellActivePredict(self, colIndex, cellIndex, timeStep,
                               activeCellsTime, predictCellsTime):
        # Check if a cell is active and was also predicting one time step before.
        cellActive = self.checkCellActive(colIndex, cellIndex, timeStep, activeCellsTime)
        cellWasPredict = self.checkCellPredict(colIndex, cellIndex, timeStep-1, predictCellsTime)
        if (cellActive and cellWasPredict):
            return True
        else:
            return False

    def updateAvgPesist(self, prevTrackingNum, cellsAvgPersistNum):
        # Update the average persistance count with an ARMA filter.
        cellsAvgPersistNum = ((1.0 - 1.0/self.delayLength) * cellsAvgPersistNum +
                              (1.0/self.delayLength) * prevTrackingNum)

    def segmentNumSynapsesActive(self, synapseMatrix, timeStep, activeCellsTime):
        # Find the number of active synapses for the previous timestep.
        # Synapses whose end is on an active cell for the timestep.
        # The synapseMatrix stores an array of synapses where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        count = 0
        for i in range(len(synapseMatrix)):
            # Make sure the synapse exist (its permanence is larger then 0)
            synpermanence = synapseMatrix[i][2]
            if synpermanence > self.connectPermanence:
                columnIndex = int(synapseMatrix[i][0])
                cellIndex = int(synapseMatrix[i][1])
                if self.checkCellActive(columnIndex, cellIndex, timeStep-1, activeCellsTime) is True:
                    count += 1

        return count

    def getBestMatchingSegment(self, cellSegmentTensor, timeStep, activeCellsTime):
        # We find the segment who was most predicting for the timestep and call
        # this the best matching segment. This means that segment has synapses whose ends are
        # connected to cells that were active in the previous timeStep.

        # The cellSegmentTensor is a 3d array holding all the segments for a particular cell.
        # For each segment there is an array of synpases and for each synapse there is an
        # array holding [columnIndex, cellIndex, permanence].

        # This routine is agressive. The permanence value of a synapse is allowed to be less
        # then connectedPermance but the number of active synapses in the segment must be
        # larger then minNumSynThreshold for the segment to be considered as active.
        # This means we need to find synapses that where previously active.
        h = 0   # mostActiveSegmentIndex
        mostActiveSegSynCount = None
        # Look through the segments for the one with the most active synapses.
        # Iterate through each segment g, for the cell i, in column c.
        for g in range(len(cellSegmentTensor)):
            # Find in the current segment, synapses that where active for the previous timeStep.
            currentSegSynCount = self.segmentNumSynapsesActive(cellSegmentTensor[g], timeStep, activeCellsTime)
            if currentSegSynCount > mostActiveSegSynCount or (mostActiveSegSynCount is None):
                h = g
                mostActiveSegSynCount = currentSegSynCount
        # Make sure the cell has at least one segment
        if len(cellSegmentTensor) > 0:
            if mostActiveSegSynCount > self.minNumSynThreshold:
                return h    # returns just the index to the
                # most active segment in the cell
        # print "returned no segment. None had enough active synapses return -1"
        return None   # Means no segment was active enough.

    def getSegmentActiveSynapses(self, segSynapseList, timeStep, activeCellsTime):
        # Find the synapse indicies of the active synapses for the timestep.
        # Synapses whose end is on an active cell for the timestep.
        # The segSynapseList stores an array of synapses where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        # Return an array of size len(segSynapseList) where a 1 indicates
        # the synapse is active and a 0 indicates inactive.
        activeSynapses = np.zeros(len(segSynapseList))
        for i in range(len(segSynapseList)):
            columnIndex = segSynapseList[i][0]
            cellIndex = segSynapseList[i][1]

            if self.checkCellActive(columnIndex, cellIndex, timeStep, activeCellsTime) is True:
                activeSynapses[i] = 1
            else:
                activeSynapses[i] = 0

        return activeSynapses

    def updateDistalSyn(self, colIndex, cellIndex, segIndex, distalSynapses, segActiveSynList):
        # Increment the synapses in the given segment that where active indicated by the list segActiveSynList.
        # DistalSynapses is a 5D tensor. Column, cell, segment, synapses, [endColumnIndex, EndCellIndex, Permanence]
        # segActiveSynList is a vector equal in length to the number of synapses in each segment.
        # Stores a 1 if that synapse in the given segment should have it's permanence value incremented.
        synapseList = distalSynapses[colIndex][cellIndex][segIndex]
        for s in range(len(synapseList)):
            # Increment the permanence of the active synapse
            if segActiveSynList[s] == 1:
                # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                # print "Incrementing syn perm [%s,%s,%s,%s]" % (c, i, segIndUpdate, s)
                distalSynapses[colIndex][cellIndex][segIndex][s][2] += self.permanenceInc
                distalSynapses[colIndex][cellIndex][segIndex][s][2] = min(1.0,
                                                                          [colIndex][cellIndex][segIndex][s][2])

    def findLeastUsedSeg(self, cellsActiveSegTimes, returnTimeStep=False):
        # Find the most unused segment from the given cells list
        # of previous active times for each segment.
        # Returns the index of the least used segment and the timeStep
        # the least used segment was last active at if required by returnTimeStep.
        leastUsedSeg = None
        oldestTime = None

        for s in range(len(cellsActiveSegTimes)):
            lastActiveTime = cellsActiveSegTimes[s]
            if ((lastActiveTime < oldestTime) or oldestTime is None):
                oldestTime = lastActiveTime
                leastUsedSeg = s
        if returnTimeStep is True:
            return leastUsedSeg, oldestTime
        else:
            return leastUsedSeg

    def newRandomPrevActiveSynapses(self, segSynList, prev2CellsActPredList, curSynapseList=None, keepConnectedSyn=False):
        # Fill the segSynList with a random selection of new synapses
        # that are connected with cells that are in the prev2CellsActPredList.
        # This list stores the cells that where in the antepenultimate active Predict state.
        # This may be from further back then 2 timesteps as the previous activePredict cells may
        # have been active for multiple timesteps.
        # Each element in the synapseList contains (colIndex, cellIndex, permanence)
        for i in range(len(segSynList)):
            # if the keepConnectedSyn option is false then create new synapses for all
            # the synapses in the segment.
            if keepConnectedSyn is False:
                if len(prev2CellsActPredList) > 0:
                    newSynEnd = random.sample(prev2CellsActPredList, 1)[0]
                    columnIndex = newSynEnd[0]
                    cellIndex = newSynEnd[1]
                    segSynList[i] = [columnIndex, cellIndex, self.newSynPermanence]
            else:
                # keepConnectedSyn is true only create new synapses by
                # overwriting weak synapses (low permanence values).
                curpermanence = curSynapseList[i][2]
                if curpermanence < self.connectPermanence:
                    #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                    if len(self.prev2CellsActPredList) > 0:
                        newSynEnd = random.sample(prev2CellsActPredList, 1)[0]
                        columnIndex = newSynEnd[0]
                        cellIndex = newSynEnd[1]
                        segSynList[i] = [columnIndex, cellIndex, self.newSynPermanence]

    def updateProximalTempPool(self, colPotInputs,
                               colActive, colPotSynPerm, timeStep):
        '''
        Update the proximal synapses (the column synapses) such that;
            a. For each currently active column increment the permanence values
               of potential synapses connected to an active input one timestep ago.
            b. For each column that was active one timestep ago increment the permanence
               values of potential synapses connected to an active input now.

        Inputs:
                1.  colPotInputs is a 2d tensor storing for each columns potential proximal
                    synapses (column synapses) whether its end is connected to an active input.
                    A 1 means it is connected to an active input 0 it's not.

                2.  colActive a 1d tensor (an array) where each element represents if a column
                    is active 1 or not 0.

                3.  colPotSynPerm is a 2d tensor storing the permanence values of every potential
                    synapse for each column.

                4.  timeStep the current time count.
        Outputs:
                1.  Updates and outputs the 2d tensor colPotSynPerm.

        '''
        for c in range(len(colActive)):
            # Iterate through each potential synpase.
            for s in range(len(colPotSynPerm[c])):
                # Update the potential synapses for the currently active columns.
                if colActive[c] == 1:
                    # If any of the columns potential synapses where connected to an
                    # active input increment the synapses permenence.
                    if self.prevColPotInputs[c][s] == 1:
                        # print "Current active Col prev input active for col, syn = %s, %s" % (c, s)
                        colPotSynPerm[c][s] += self.spatialPermanenceInc
                        colPotSynPerm[c][s] = min(1.0, colPotSynPerm[c][s])
                # Update the potential synapses for the previous active columns.
                if self.prevColActive[c] == 1:
                    # If any of the columns potential synapses are connected to a
                    # currently active input increment the synapses permenence.
                    if colPotInputs[c][s] == 1:
                        # print "Prev active Col current input active for col, syn = %s, %s" % (c, s)
                        colPotSynPerm[c][s] += self.spatialPermanenceInc
                        colPotSynPerm[c][s] = min(1.0, colPotSynPerm[c][s])

        # Store the current inputs to the potentialSynapses to use next time.
        self.prevColPotInputs = colPotInputs
        self.prevColActive = colActive

        return colPotSynPerm

    def updateDistalTempPool(self, timeStep, predictCellsTime, activeCellsTime, activeSeg, distalSynapses):
        '''
        Update the distal synapses (the cell synapses) such that;

        Inputs:
                1.  timeStep is the number of iterations that the HTM has been through.
                    It is just an incrementing integer used to keep track of time.

                3.  "predictCellsTime" This 3D tensor is returned by this function. It is the timeSteps when cells where
                    in the predictive state last. The 1st dimension stores the columns the 2nd is the cells in the columns.
                    Each element stores the last 2 timestep when this cell was in the predictive state.

                4.  activeCellsTime is a 3D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the last two timeSteps when the cell was in an active State.
                    It must have the dimesions of self.numColumns * self.cellsPerColumn * 2.

                5.  activeSeg is a 3D tensor. The first dimension is the columns, the second the cells and the 3rd is
                    the segment in the cells. For each segment a timeStep is stored indicating when the segment was
                    last in an active state. This means it was predicting that the cell would become active in the
                    next timeStep. This is what the CLA paper calls a "SEQUENCE SEGMENT".

                6.  distalSynapses is a 5D tensor. The first dimension stores the columns, the 2nd is the cells
                    in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
                    segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
                    This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
                    It does not change size. Its size is fixed when this class is constructed.

        Updates:
                1. self.cellsTrackingNum
                2. self.cellsAvgPersist
                3. self.cellsPersistance
                4. predictCellsTime

        '''
        for c in range(self.numColumns):
            for i in range(self.cellsPerColumn):
                if (self.checkCellActivePredict(c, i, timeStep,
                                                activeCellsTime, predictCellsTime) is True):
                    # Increment the tracking number for the active predict cell
                    self.cellsTrackingNum[c][i] += 1
                else:
                    # Update the avg persistence count for the cell
                    if ((self.cellsTrackingNum[c][i] > 0) and self.checkCellActive(c, i, timeStep, activeCellsTime)):
                        self.updateAvgPesist(self.cellsTrackingNum[c][i], self.cellsAvgPersist[c][i])
                    # Reset the tracking number back to zero
                    self.cellsTrackingNum[c][i] = 0
                # Calculate the persistance value
                self.cellsPersistance[c][i] = self.cellsAvgPersist[c][i] - self.cellsTrackingNum[c][i]
                # limit the persistance count
                if self.cellsPersistance[c][i] < 0:
                    self.cellsPersistance[c][i] = 0
                # If the cell was previously predicting or active and the persistance value is larger then zero
                # and it is no longer predictng then keep the cell in the predicting state.
                if ((self.checkCellPredict(c, i, timeStep-1, predictCellsTime) or
                     (self.checkCellActive(c, i, timeStep-1, activeCellsTime))) and
                    (self.cellsPersistance[c][i] > 0) and
                   (self.checkCellPredict(c, i, timeStep, predictCellsTime) is False)):
                    self.setPredictCell(c, i, timeStep, predictCellsTime)
                # If the cell is now active predict (active and was predicting) then
                # add new distal synapses that connect to the learn cells at t-1 timesteps ago.
                if (self.checkCellActivePredict(c, i, timeStep, activeCellsTime, predictCellsTime)):
                    # Find the segment which already contains enough synapses to have predicted 1 timesteps ago.
                    h = self.getBestMatchingSegment(distalSynapses[c][i], timeStep-1, activeCellsTime)
                    if h is not None:
                        # Find the synapses that where active and increment their permanence values.
                        segActiveSynList = self.getSegmentActiveSynapses(distalSynapses[c][i][h], timeStep, activeCellsTime)
                        self.updateDistalSyn(c, i, h, distalSynapses, segActiveSynList)
                    else:
                        # No best matching segment was found so a new segment will be
                        # created overwrite the least used segment.
                        h, lastTimeStep = self.findLeastUsedSeg(activeSeg[c][i], True)
                        print "new Random Seg created for c,i,h = %s, %s, %s" % (c, i, h)

                        # TODO
                        # Implement a self.prev2CellsActPredList keeping track of the antepenultimate
                        # activePredict cells.
                        self.newRandomPrevActiveSynapses(distalSynapses[c][i][h], self.prev2CellsActPredList)

        return distalSynapses


def runTempPoolUpdateProximal(tempPooler, colPotInputs, colPotSynPerm, timeStep):
    # Run the temporal poolers function to update the proximal synapses for a test.
    print "INITIAL colPotSynPerm = \n%s" % colPotSynPerm
    print "colPotInputs = \n%s" % colPotInputs
    print "colActive = \n%s" % colActive
    # Run through calculator
    test_iterations = 4
    for i in range(test_iterations):
        timeStep += 1
        colPotSynPerm = tempPooler.updateProximalTempPool(colPotInputs,
                                                          colActive,
                                                          colPotSynPerm,
                                                          timeStep
                                                          )
        print "colPotSynPerm = \n%s" % colPotSynPerm


def runTempPoolUpdateDistal(tempPooler, timeStep, predictCellsTime,
                            activeCellsTime, activeSeg, distalSynapses):
    # Run the temporal poolers function to update the distal synapses for a test.
    print "INITIAL distalSynapses = \n%s" % distalSynapses

    # Run through calculator
    test_iterations = 1
    for i in range(test_iterations):
        timeStep += 1
        distalSynapses = tempPooler.updateDistalTempPool(timeStep,
                                                         predictCellsTime, activeCellsTime,
                                                         activeSeg, distalSynapses)
        print "distalSynapses = \n%s" % distalSynapses

if __name__ == '__main__':

    numRows = 4
    numCols = 4
    numColumns = numRows * numCols
    maxSegPerCell = 1
    maxSynPerSeg = 2
    cellsPerColumn = 2
    spatialPermanenceInc = 1.0
    spatialPermanenceDec = 0.2
    seqPermanenceInc = 0.1
    seqPermanenceDec = 0.02
    newSynPermanence = 0.3
    minNumSynThreshold = 1
    connectPermanence = 0.2
    maxNumTempPoolPatterns = 3
    activeColPermanenceDec = float(spatialPermanenceInc)/float(maxNumTempPoolPatterns)
    potentialWidth = 2
    potentialHeight = 2
    numPotSyn = 4
    tempDelayLength = 4
    timeStep = 1
    numColumns = numRows * numCols
    # Create an array representing the permanences of colums synapses
    colPotSynPerm = np.random.rand(numColumns, numPotSyn)
    # Create an array representing the potential inputs to each column
    colPotInputs = np.random.randint(2, size=(numColumns, numPotSyn))
    # Create an array representing the active columns
    colActive = np.random.randint(2, size=(numColumns))

    # Create the distalSynapse 5d tensor holding the information of the distal synapses.
    distalSynapses = np.zeros((numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, 3))
    for index, x in np.ndenumerate(distalSynapses):
        # print index, x
        if index[4] == 2:
            distalSynapses[index] = random.randint(0, 10) / 10.0
        if index[4] == 1:
            distalSynapses[index] = random.randint(0, cellsPerColumn-1)
        if index[4] == 0:
            distalSynapses[index] = random.randint(0, numColumns-1)

    # Create the predictive cells times
    predictCellsTime = np.zeros((numColumns, cellsPerColumn, 2))
    # Create the active cells times
    activeCellsTime = np.zeros((numColumns, cellsPerColumn, 2))
    activeSegsTime = np.zeros((numColumns, cellsPerColumn, maxSegPerCell))

    # # Profile and save results as a picture
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'np_temporal.png'
    # with PyCallGraph(output=graphviz):

    tempPooler = TemporalPoolCalculator(cellsPerColumn, numColumns, numPotSyn,
                                        spatialPermanenceInc, spatialPermanenceDec,
                                        seqPermanenceInc, seqPermanenceDec,
                                        minNumSynThreshold, newSynPermanence,
                                        connectPermanence, tempDelayLength)

    # Test the temporal poolers update proximal synapse function
    # runTempPoolUpdateProximal(tempPooler, colPotInputs, colPotSynPerm, timeStep)

    # Test the temporal poolers update distal synapse function
    runTempPoolUpdateDistal(tempPooler, timeStep, predictCellsTime,
                            activeCellsTime, activeSegsTime, distalSynapses)


