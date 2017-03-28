import cProfile
import random
import numpy as np

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

'''
A class to calculate the temporal pooling for a HTM layer.


'''

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


class TemporalPoolCalculator():
    def __init__(self, cellsPerColumn, numColumns, numPotSyn,
                 spatialPermanenceInc, 
                 seqPermanenceInc, 
                 minNumSynThreshold, newSynPermanence,
                 connectPermanence, delayLength):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        self.numPotSynapses = numPotSyn
        # The value by which the spatial poolers synapses (proximal synapses)
        # permanence values change by.
        self.spatialPermanenceInc = spatialPermanenceInc
        # The value by which the sequence poolers synapses (distal synapses)
        # permanence values change by.
        self.seqPermanenceInc = seqPermanenceInc
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
        # The timeSteps when cells first enetered the learning state last. This doesn't include subsequent timesteps
        # where the cell stayed in the learning state.
        # This is a 3D tensor. The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timesteps when this cell first enetered the learning state last.
        self.newLearnCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])

    def checkCellLearn(self, colIndex, cellIndex, timeStep, learnCellsTime):
        # Check if the given cell was learning at the timestep given.
        # We need to check the learnCellsTime tensor which holds multiple
        # previous timeSteps when each cell was last in the learn state.
        if learnCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if learnCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def checkCellActive(self, colIndex, cellIndex, timeStep, activeCellsTime):
        # Check if the given cell was active at the timestep given.
        # We need to check the activeCellsTime tensor which holds multiple
        # previous timeSteps when each cell was last active.
        # import ipdb; ipdb.set_trace()

        if activeCellsTime[int(colIndex)][int(cellIndex)][0] == timeStep:
            return True
        if activeCellsTime[int(colIndex)][int(cellIndex)][1] == timeStep:
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

    def setLearnCell(self, colIndex, cellIndex, timeStep, learnCellsTime):
        # Set the given cell at colIndex, cellIndex into a learn state for the
        # given timeStep.
        # We need to check the learnCellsTime tensor which holds multiple
        # previous timeSteps and set the oldest one to the given timeStep.
        if learnCellsTime[colIndex][cellIndex][0] <= learnCellsTime[colIndex][cellIndex][1]:
            learnCellsTime[colIndex][cellIndex][0] = timeStep
        else:
            learnCellsTime[colIndex][cellIndex][1] = timeStep

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

    def checkColBursting(self, colIndex, timeStep, activeCellsTime):
        # Check if the given column is bursting or not for a particular timeStep.
        cellsActive = 0
        for cellIndex in range(len(activeCellsTime[colIndex])):
            # Count the number of cells in the column that where active.
            if self.checkCellActive(colIndex, cellIndex, timeStep, activeCellsTime) is True:
                cellsActive += 1
            if cellsActive > 1:
                break
        if cellsActive == 1:
            return True
        else:
            return False 

    def updateAvgPesist(self, prevTrackingNum, cellsAvgPersistNum):
        # Update the average persistance count with an ARMA filter.
        cellsAvgPersistNum = ((1.0 - 1.0/self.delayLength) * cellsAvgPersistNum +
                              (1.0/self.delayLength) * prevTrackingNum)

    def segmentNumSynapsesActive(self, synapseMatrix, potPrev2LearnCellsList):
        # Find the number of synapses whose ends connect to the cells in
        # the list of antepenultimate cells potPrev2LearnCellsList.
        # Note: the input potPrev2LearnCellsList stores the antepenultimate learning cells as
        # [[columnIndex, cellIndex], ...]

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
                # TODO
                # Make a faster method of searching the potPrev2LearnCellsList
                for colCellInd in potPrev2LearnCellsList:
                    if (colCellInd[0] == columnIndex and colCellInd[1] == cellIndex):
                        count += 1

        return count

    def getBestMatchingSegment(self, cellSegmentTensor, potPrev2LearnCellsList):
        # We find the segment who was most predicting and call
        # this the best matching segment. This means that segment has synapses whose ends are
        # connected to cells that are from the antepenultimate list of learning cells.
        # Note: the input potPrev2LearnCellsList stores the antepenultimate learning cells as
        # [[columnIndex, cellIndex], ...]

        # The cellSegmentTensor is a 3d array holding all the segments for a particular cell.
        # For each segment there is an array of synpases and for each synapse there is an
        # array holding [columnIndex, cellIndex, permanence].

        # This routine is agressive. The permanence value of a synapse is allowed to be less
        # then connectedPermance but the number of active synapses in the segment must be
        # larger then minNumSynThreshold for the segment to be considered as active.

        h = 0   # mostActiveSegmentIndex
        mostActiveSegSynCount = None
        # Look through the segments for the one with the most active synapses.
        # Iterate through each segment g, for the cell i, in column c.
        for g in range(len(cellSegmentTensor)):
            # Find in the current segment, synapses that where in potPrev2LearnCellsList.
            currentSegSynCount = self.segmentNumSynapsesActive(cellSegmentTensor[g], potPrev2LearnCellsList)
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
                distalSynapses[colIndex][cellIndex][segIndex][s][2] += self.seqPermanenceInc
                distalSynapses[colIndex][cellIndex][segIndex][s][2] = min(1.0,
                                                                          distalSynapses[colIndex][cellIndex][segIndex][s][2])

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

    def getPrev2NewLearnCells(self, timeStep, newLearnCellsList, learnCellsTime, activeCellsTime, numCells):
        # Find from the newLearnCellsTime tensor the cells that most recently entered the learning state
        # but were not active in the previous timestep. Note we have not updated the newLearnCellsTime tensor yet from
        # the last timeStep so no cells that just entered the learning state this timestep will be included either.
        # Return at least the number of cells specified from the input numCells unless not enough cells have
        # entered the learn state yet. The returned list holds [columnIndex, cellIndex] for each element.
        prev2CellsActPredList = []
        # Sort the newLearnCellsTime and then start from the latest cell checking if it was active in the previous timestep.
        # If it wasn't add it to the prev2CellsActPredList. Keep going until you have more then numCells added to the list or
        # the rest of the cells never entered the learn state (the timestep is -1). If you have added numCells to the
        # prev2CellsActPredList then check what the current timeStep is and add the rest of the cells with this
        # timestep as well.
        # We need the column and cell indicies not the timesteps so use argsort instead of sort.
        # Create a 1d array to sort then return the indicies.
        sortFlatNewLearnCellsTime = np.argsort(self.newLearnCellsTime.ravel())
        # Convert the ordered 1d array of indices into a tuple of coordinate arrays.
        # Each element represnets the indices for positions in self.newLearnCellsTime.
        # The indices are sorted from longest ago to most recent.
        # Remember that the newLearnCellsTime stored the last 2 timesteps for each cell in each column.
        sortFlatNewLearnCellsTime = np.dstack(np.unravel_index(sortFlatNewLearnCellsTime,
                                                               (self.numColumns, self.cellsPerColumn, 2)))
        sortFlatNewLearnCellsTime = sortFlatNewLearnCellsTime[0]

        foundNumCells = False
        latestTimeStep = None
        for index3D in reversed(sortFlatNewLearnCellsTime):
            columnIndex = index3D[0]
            cellIndex = index3D[1]
            learnCellsTimeStep = self.newLearnCellsTime[index3D[0]][index3D[1]][index3D[2]]
            #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

            if (foundNumCells is False) or (foundNumCells is True and learnCellsTimeStep == latestTimeStep):
                if learnCellsTimeStep >= 0:
                    if self.checkCellActive(columnIndex, cellIndex, timeStep-1, activeCellsTime) is False:
                        prev2CellsActPredList.append([columnIndex, cellIndex])
                if len(prev2CellsActPredList) == numCells:
                    foundNumCells = True
                    latestTimeStep = learnCellsTimeStep
            if (foundNumCells is True and learnCellsTimeStep < latestTimeStep):
                # Break the for loop we don't need to continue searching for cells to add to the list
                break

        # From the list of cells that are in the learn state newLearnCellsList, update the newLearnCells tensor.
        # This tensor stores the timesteps when a cell first enters the learn state not subsequent timesteps.
        # Check the learnCellsTime tensor to see if this is the first time a cell in the newLearnCellsList
        # entered the learning state.
        for j in range(len(newLearnCellsList)):
            colInd = newLearnCellsList[j][0]
            cellInd = newLearnCellsList[j][1]
            #print "Checking if learning at timeStep = %s, learnCellsTime[%s][%s] = %s" % (timeStep-1, colInd, cellInd, learnCellsTime[colInd][cellInd])
            if self.checkCellLearn(colInd, cellInd, timeStep-1, learnCellsTime) is False:
                #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                self.setLearnCell(colInd, cellInd, timeStep, self.newLearnCellsTime)

        return prev2CellsActPredList

    def newRandomPrevActiveSynapses(self, segSynList, prev2CellsActPredList, curSynapseList=None, keepConnectedSyn=False):
        # Fill the segSynList with a random selection of new synapses
        # that are connected with cells that are in the prev2CellsActPredList.
        # This list stores the cells that where in the antepenultimate learning state.
        # This may be from further back then 2 timesteps as the previous learning cells may
        # have been active for multiple timesteps.
        # Each element in the synapseList contains (colIndex, cellIndex)
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

    #@do_cprofile  # For profiling
    def updateProximalTempPool(self, colPotInputs,
                               colActive, colPotSynPerm, timeStep, activeCellsTime):
        '''
        Update the proximal synapses (the column synapses) such that;
            a. For each currently active column increment the permanence values
               of potential synapses connected to an active input one timestep ago. 
               Do not do this for any bursting columns.
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

                5.  "activeCellsTime" This is a 3D tensor. It is the timeSteps when cells where
                    active last. The 1st dimension stores the columns the 2nd is the cells in the columns.
                    Each element stores the last 2 timestep when this cell was active last.
        Outputs:
                1.  Updates and outputs the 2d tensor colPotSynPerm.

        '''

        # Only perform temporal proximal pooling if the proximal permanence increment value is larger then zero!
        if self.spatialPermanenceInc > 0.0:
            for c in range(len(colActive)):
                # Update the potential synapses for the currently active columns.
                if colActive[c] == 1:
                    # Iterate through each potential synpase.
                    for s in range(len(colPotSynPerm[c])):
                        # Check to make sure the column isn't bursting.
                        if self.checkColBursting(c, timeStep, activeCellsTime) is True:
                            # If any of the columns potential synapses where connected to an
                            # active input increment the synapses permenence.
                            if self.prevColPotInputs[c][s] == 1:
                                # print "Current active Col prev input active for col, syn = %s, %s" % (c, s)
                                colPotSynPerm[c][s] += self.spatialPermanenceInc
                                colPotSynPerm[c][s] = min(1.0, colPotSynPerm[c][s])
                        # Update the potential synapses for the previous active columns.
                        if self.prevColActive[c] == 1:
                            # Check to make sure the column wasn't bursting.
                            if self.checkColBursting(c, timeStep, activeCellsTime) is True:
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

    def updateDistalTempPool(self, timeStep, newLearnCellsList, learnCellsTime, predictCellsTime,
                             activeCellsTime, activeSeg, distalSynapses):
        '''
        Update the distal synapses (the cell synapses) such that;

        Inputs:
                1.  timeStep is the number of iterations that the HTM has been through.
                    It is just an incrementing integer used to keep track of time.

                2.  "newLearnCellsList" is a 2d tensor storing the cells that are in the learn state in each column
                    for the current timeStep. It is a variable length 2d array storing the columnIndex and cell index
                    of cells currently in the learn state.

                3. "learnCellsTime" This 3D tensor is returned by this function. It is the timeSteps when cells where
                    in the learning state last. The 1st dimension stores the columns the 2nd is the cells in the columns.
                    Each element stores the last 2 timestep when this cell was in the learning state.

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

        # Only perform temporal distal pooling if the distal permanence increment value is larger then zero!
        if self.seqPermanenceInc > 0.0:
            # Get a list of the antepenultimate new learning cells.
            # These are the cells that were last in the learning state but were not
            # active in the previous timestep or current timestep.
            # The list needs to be at least as big as the num synpases in a segment since the segment will
            # be filled with entirely new synapses connected to cells from this list.
            numCellsNeeded = len(distalSynapses[0][0][0])
            potPrev2LearnCellsList = self.getPrev2NewLearnCells(timeStep, newLearnCellsList,
                                                                learnCellsTime, activeCellsTime, numCellsNeeded)

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
                    # add new distal synapses that connect to the antepenultimate learning cells.
                    if (self.checkCellActivePredict(c, i, timeStep, activeCellsTime, predictCellsTime)):
                        # Find the segment which already contains enough synapses to have predicted that it would be
                        # in the active predict state now, use the antepenultimate learning state cells.
                        # This may be from further back then 2 timesteps as the previous learning cells may
                        # have been active for multiple timesteps.
                        h = self.getBestMatchingSegment(distalSynapses[c][i], potPrev2LearnCellsList)
                        if h is not None:
                            # Find the synapses that where active and increment their permanence values.
                            segActiveSynList = self.getSegmentActiveSynapses(distalSynapses[c][i][h], timeStep, activeCellsTime)
                            self.updateDistalSyn(c, i, h, distalSynapses, segActiveSynList)
                        else:
                            # No best matching segment was found so a new segment will be
                            # created overwrite the least used segment.
                            h, lastTimeStep = self.findLeastUsedSeg(activeSeg[c][i], True)
                            # print "new Random Seg created for c,i,h = %s, %s, %s" % (c, i, h)
                            self.newRandomPrevActiveSynapses(distalSynapses[c][i][h], potPrev2LearnCellsList)

        return distalSynapses


def runTempPoolUpdateProximal(tempPooler, colPotInputs, colPotSynPerm, timeStep, activeCellsTime):
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
                                                          timeStep,
                                                          activeCellsTime
                                                          )
        print "colPotSynPerm = \n%s" % colPotSynPerm


def runTempPoolUpdateDistal(tempPooler, timeStep, newLearnCellsList, learnCellsTime, predictCellsTime,
                            activeCellsTime, activeSeg, distalSynapses):
    # Run the temporal poolers function to update the distal synapses for a test.
    print "INITIAL distalSynapses = \n%s" % distalSynapses

    # Run through calculator
    test_iterations = 1
    for i in range(test_iterations):
        timeStep += 1
        distalSynapses = tempPooler.updateDistalTempPool(timeStep, newLearnCellsList, learnCellsTime,
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
    seqPermanenceInc = 0.1
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

    # Create a list sotring which cells are in the learning state for the current timestep [[colInd, CellInd], ...]
    newLearnCellsList = []
    # Create the learning cells times
    learnCellsTime = np.zeros((numColumns, cellsPerColumn, 2))
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
                                        spatialPermanenceInc, 
                                        seqPermanenceInc, 
                                        minNumSynThreshold, newSynPermanence,
                                        connectPermanence, tempDelayLength)

    # Test the temporal poolers update proximal synapse function
    # runTempPoolUpdateProximal(tempPooler, colPotInputs, colPotSynPerm, timeStep, activeCellsTime)

    # Test the temporal poolers update distal synapse function
    runTempPoolUpdateDistal(tempPooler, timeStep, newLearnCellsList, learnCellsTime, predictCellsTime,
                            activeCellsTime, activeSegsTime, distalSynapses)


