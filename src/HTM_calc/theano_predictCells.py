import numpy as np
import math
import random
import theano.tensor as T
from theano import function
from theano import scan

import cProfile
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

'''
A class used to set cells in a predictive state

THIS CLASS IS A REIMPLEMENTATION OF THE ORIGINAL CODE:
    """
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
    """
'''


class predictCellsCalculator():
    def __init__(self, numColumns, cellsPerColumn, maxSegPerCell, maxSynPerSeg, connectPermanence, activationThreshold):
        self.numColumns = numColumns
        self.cellsPerColumn = cellsPerColumn
        # Maximum number of segments per cell
        self.maxSegPerCell = maxSegPerCell
        # Maximum number of synapses per segment
        self.maxSynPerSeg = maxSynPerSeg
        # The minimum required permanence value required by a synapse for it
        # to be connected.
        self.connectPermanence = connectPermanence
        # More than this many synapses on a segment must be active for
        # the segment to be active.
        self.activationThreshold = activationThreshold
        # A tensor storing for each cells segments the number of synapses connected to active cells.
        self.currentSegSynCount = np.zeros((self.numColumns,
                                            self.cellsPerColumn,
                                            maxSegPerCell))
        # The timeSteps when cells where in the predicitive state last. This is a 3D tensor.
        # The 1st dimension stores the columns the 2nd is the cells in the columns.
        # Each element stores the last 2 timestep when this cell was predicitng last.
        self.predictCellsTime = np.array([[[-1, -1] for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 2d tensor for each cell holds [segIndex] indicating which segment to update.
        # If the index is -1 this means don't create a new segment.
        self.segIndUpdate = np.array([[-1 for x in range(self.cellsPerColumn)] for y in range(self.numColumns)])
        # A 3d tensor for each cell holds a synpase list indicating which
        # synapses in the segment (the segment index is stored in the segIndUpdate tensor)
        # are active [activeSynList 0 or 1].
        self.segActiveSyn = np.array([[[-1 for z in range(self.maxSynPerSeg)]
                                     for x in range(self.cellsPerColumn)]
                                     for y in range(self.numColumns)])

        # activeSegsTime is a 3D tensor. The first dimension is the columns, the second the
        # cells and the 3rd is the segment in the cells. For each segment a timeStep is stored
        # indicating when the segment was last in an active state. This means it was
        # predicting that the cell would become active in the next timeStep.
        # This is what the CLA paper calls a "SEQUENCE SEGMENT".
        self.activeSegsTime = np.zeros((self.numColumns, self.cellsPerColumn, self.maxSegPerCell))

        # Store for every segment the number of distal synapses that have permanence values greater then
        # the connectPermanence and are connected to cells that are active.
        self.predictionLevel = np.array([[[0.0 for k in range(self.maxSegPerCell)]
                                         for i in range(self.cellsPerColumn)]
                                         for j in range(self.numColumns)])

        # Store for each segment in each cell in each column the number of synpases that are connected
        # (permanence is larger then the connectPermanence value) and are active (their end is connected to
        # an active cell). This is a 3d tensor.
        self.segConActiveSynCount = None

        # Create theano variables and functions
        ############################################

        # Create the theano function for calculating
        # the multiplication elementwise of 2 matricies.
        self.i_grid = T.matrix(dtype='float32')
        self.j_grid = T.matrix(dtype='float32')
        self.multi_vals = self.i_grid * self.j_grid
        self.multi_grids = function([self.i_grid, self.j_grid],
                                    self.multi_vals,
                                    on_unused_input='warn',
                                    allow_input_downcast=True)

        # Create the theano function for finding the count of active connected distal
        # synapses for each segment.
        # Calculate for every synapse from the distalSynapses Tensor an output 4D tensor which 
        # contains a one if that syapses is connected and active or zero if not.
        self.time_step1 = T.scalar(dtype='int32')
        self.con_perm1 = T.scalar(dtype='float32')
        self.act_cellTimes = T.tensor3(dtype='float32')
        # Create a 5d tensor type in theano
        dtensor5 = T.TensorType('float32', (False,)*5)
        #self.distal_syn = dtensor5()
        self.distal_syn_colInd = T.tensor4(dtype='float32')
        self.distal_syn_cellInd = T.tensor4(dtype='float32')
        self.distal_syn_perm =  T.tensor4(dtype='float32')
        # Calcualte the connected synapses (their permanence value is greater then the connected permanence parameter).
        self.syn_con = T.switch(T.ge(self.distal_syn_perm, self.con_perm1), 1, 0)
        # Calculate the synpases which connect to currently active cells. This returns 2 elements for each synapse since the cell_times 
        # contains the last two time step the cell was active and both these are checked.
        self.syn_actnum = T.switch(T.eq(self.act_cellTimes[T.cast(self.distal_syn_colInd, 'int32'), 
                                                           T.cast(self.distal_syn_cellInd, 'int32')], 
                                        self.time_step1),
                                   1, 0)
        # If a synapse connected to an active cell then check its permanence value. 
        self.syn_act = T.switch(T.gt(self.syn_actnum.sum(axis=4),0), self.syn_con, 0)
        # Sum all the synpases which returned a one. This is the count of the active connected synapses for each segment.
        self.seg_actcon_count = self.syn_act.sum(axis=3)
        self.connectActSynCount = function([self.distal_syn_colInd, 
                                            self.distal_syn_cellInd,
                                            self.distal_syn_perm, 
                                            self.time_step1, 
                                            self.act_cellTimes,
                                            self.con_perm1],
                                            self.seg_actcon_count,
                                            allow_input_downcast=True)

        # Create the theano function for finding if each segments count is larger
        # then the self.activationThreshold. If so update the given tensor "activeSegsTime"
        # by setting that segments time step to the current time.
        self.time_step2 = T.scalar(dtype='int32')
        self.act_segTimes = T.tensor3(dtype='float32')
        self.con_actSynCount = T.tensor3(dtype='float32')
        self.act_thresh = T.scalar(dtype='float32')
        self.seg_activated = T.switch(T.gt(self.con_actSynCount, self.act_thresh), self.time_step2, self.act_segTimes)
        self.updateActSegTimes = function([self.con_actSynCount,
                                           self.act_segTimes,
                                           self.time_step2, 
                                           self.act_thresh],
                                           self.seg_activated,
                                           allow_input_downcast=True)

        # Create the theano funciton for finding the most predicting segment in each column.
        # return a tensor storing for each column; 
        #   [[mostPredSegmentInd],
        #    [mostPredCellInd], 
        #    [columnPredicting]] 
        # Eg. The returned tensor has at position 0 an array of indicies. Each element in this array
        #     corresponds to one column and stores the index of the segment in a cell in that column that
        #     is most predicting.
        #     The returned tensor has at position 1 an array of indicies. Each element in this array
        #     corresponds to one column and stores the index of the cell in that column that
        #     is most predicting.
        #     The returned tensor has at position 2 an array of bools. Each element in this array
        #     corresponds to one column and stores whether that column is predicting or not (1 or 0).
        # Note: If 2 cells in a column or 2 segments in a cell have the same count for the active and connected
        #       distal synapses then the first one (the one with the lower cell or segment index) is returned as
        #       the most predicting cell and segment.

        self.seg_actcon_count2 = T.tensor3(dtype='float32')
        self.act_thresh2 = T.scalar(dtype='float32')
        self.most_predSegInCell = T.max_and_argmax(self.seg_actcon_count2, axis=2, keepdims=False)
        self.most_predSegInCellVal = T.max(self.seg_actcon_count2, axis=2, keepdims=False)
        self.most_predSegSegInd = T.argmax(self.seg_actcon_count2, axis=2, keepdims=False)
        
        self.most_predSegInCol = T.max_and_argmax(self.most_predSegInCell[0], axis=1, keepdims=False)
        self.col_pred = T.switch(T.gt(self.most_predSegInCol[0], self.act_thresh2), 1, 0)
        self.most_predSegCountInd = T.cast(T.argmax(self.most_predSegInCellVal, axis=1, keepdims=False), 'int32')

        # Use a theano scan function to loop through the tensor most_predSegCountInd and most_predSegSegInd.
        # This function returns the mostPredSegmentInd (the index of the most active segment in each column).
        self.i = T.iscalar('i') #Number of iterations.
        self.most_predSegInd, updates = scan(fn=self.get_bwa,
                                                    outputs_info=None,
                                                    non_sequences=[self.most_predSegCountInd, self.most_predSegSegInd],
                                                    sequences=T.arange(self.i),
                                                    n_steps = self.numColumns)
        # Return [[mostPredSegmentInd], [mostPredCellInd], [columnPredicting]]
        self.most_predCellInd = T.argmax(self.most_predSegInCell[0], axis=1, keepdims=False)
        self.most_pred = T.as_tensor_variable([self.most_predSegInd, self.most_predCellInd, self.col_pred])
        self.mostPredSegInfo = function([self.seg_actcon_count2,
                                         self.act_thresh2,
                                         self.i],
                                         self.most_pred,
                                         allow_input_downcast=True,
                                         on_unused_input='ignore')


        #### END of Theano functions and variables definitions
        #################################################################
        # The folowing variables are used for indicies when looking up values
        # in matricies from within a theano function.
        # Create a matrix that just holds the column index for each column.
        self.col_numMat = np.array([i for i in range(self.numColumns)])

    def get_bwa(self, i, avec, bmat):
        # A function use din a Theano loop from the theano function mostPredSegInfo
        return bmat[i, avec[i]]
        
    def getActiveSegTimes(self):
        # Return the activeSegsTime tensor which holds only the most recent
        # timeSteps that each segment was active for.
        return self.activeSegsTime

    def getSegConActiveSynCount(self):
        # Return for each segment in each cell in each column the number of synpases that are connected and active.
        return self.segConActiveSynCount

    def getSegUpdates(self):
        # Return the tensors storing information on which distal synapses
        # in which segments learning should be performed on.
        return self.segIndUpdate, self.segActiveSyn

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
            if segSynapseList[i][2] >= self.connectPermanence:
                if self.checkCellActive(columnIndex, cellIndex, timeStep, activeCellsTime) is True:
                    activeSynapses[i] = 1
                else:
                    activeSynapses[i] = 0
            else:
                activeSynapses[i] = 0

        return activeSynapses

    def checkCellPredicting(self, colIndex, cellIndex, timeStep):
        # Check if the given cell was in the predicitng state at the timestep given.
        # We check the 3D predictingCellsTensor tensor which holds multiple
        # previous timeSteps when each cell was last in the predicitng state.
        if self.predictCellsTime[colIndex][cellIndex][0] == timeStep:
            return True
        if self.predictCellsTime[colIndex][cellIndex][1] == timeStep:
            return True
        return False

    def setPredictCell(self, colIndex, cellIndex, timeStep):
        # Set the given cell at colIndex, cellIndex into a predictive state for the
        # given timeStep.
        # We need to check the predictCellsTime tensor which holds multiple
        # previous timeSteps and set the oldest one to the given timeStep.
        if self.predictCellsTime[colIndex][cellIndex][0] <= self.predictCellsTime[colIndex][cellIndex][1]:
            self.predictCellsTime[colIndex][cellIndex][0] = timeStep
        else:
            self.predictCellsTime[colIndex][cellIndex][1] = timeStep

    def setActiveSeg(self, colIndex, cellIndex, segIndex, timeStep):
        # Set the given segment at colIndex, cellIndex, segInde into an active state for the
        # given timeStep.
        # We need to check the activeSegsTime tensor which holds only the most recent
        # timeSteps that each segment was active for.
        self.activeSegsTime[colIndex][cellIndex][segIndex] = timeStep

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

    def segmentNumSynapsesActive(self, synapseMatrix, timeStep, activeCellsTime):
        # Find the number of active synapses for the current timestep.
        # Synapses whose end is on an active cell for the current timestep.
        # The synapseMatrix stores an array of synpases where
        # each synapse stores the end connection of that synpase.
        # The end of the synapse connects to a cell in a column.
        # [columnIndex, cellIndex, permanence]
        # Also make sure the synapse has a permanence larger then the minimum
        # connected value.
        count = 0
        for i in range(len(synapseMatrix)):
            if synapseMatrix[i][2] > self.connectPermanence:
                columnIndex = int(synapseMatrix[i][0])
                cellIndex = int(synapseMatrix[i][1])
                if self.checkCellActive(columnIndex, cellIndex, timeStep, activeCellsTime) == True:
                    count += 1

        return count

    #@do_cprofile  # For profiling
    def updatePredictiveState(self, timeStep, activeCellsTime, distalSynapses):
    	'''
        This function calculates which cells should be set into the predictive state.
        The predictive state is when a cell is predicting it will be active on the
        next timestep becuase it currently has enough synpases in a segment group
        which connect to currently active cells.

        Inputs:
                1.  timeStep is the number of iterations that the HTM has been through.
                    It is just an incrementing integer used to keep track of time.

                2.  activeCellsTime is a 3D tensor. The first dimension stores the columns the second is the cells
                    in the columns. Each cell stores the last two timeSteps when the cell was in an active State.
                    It must have the dimesions of self.numColumns * self.cellsPerColumn * 2.

                3.  distalSynapses is a 5D tensor. The first dimension stores the columns, the 2nd is the cells
                    in the columns, 3rd stores the segments for each cell, 4th stores the synapses in each
                    segment and the 5th stores the end connection of the synapse (column number, cell number, permanence).
                    This tensor has a size of numberColumns * numCellsPerCol * maxNumSegmentsPerCell * maxNumSynPerSeg.
                    It does not change size. Its size is fixed when this class is constructed.

        Updates:
                1.  "predictCellsTime" This 3D tensor is returned by this function. It is the timeSteps when cells where
                    in the predictive state last. The 1st dimension stores the columns the 2nd is the cells in the columns.
                    Each element stores the last 2 timestep when this cell was in the predictive state.

                2. Two tensors storing information on which segments to update for a cell.
                   The two tensors are needed as one stores the segment index that is to be updated
                   and the other stores which synpases where active when the segment index was calcualted.
                   A cell can only store information about updating one segment at a time.
                   The two tensors are outlined below, none of them change size.

                     a. A 2D tensor "segIndUpdate" for each cell holds [segIndex] indicating which segment to update.
                        If the index is -1 don't update any segments.
                     b. A 3D tensor "segActiveSyn" for each cell holds a synpase list indicating which
                        synapses in the segment (the segment index is stored in the segIndUpdate tensor)
                        are active [activeSynList 0 or 1].

                3. "activeSegsTime" is a 3D tensor. The first dimension is the columns, the second the cells and the 3rd is
                    the segment in the cells. For each segment a timeStep is stored indicating when the segment was
                    last in an active state. This means it was predicting that the cell would become active in the
                    next timeStep. This is what the CLA paper calls a "SEQUENCE SEGMENT".
                    This tensor is updated on each call of this function.

        '''
        # THEANO IMPLEMENTATION
        '''
            1. Calculate for every synapse from the distalSynapses Tensor an output 4D tensor which 
               contains a one if that syapses is connected and zero if not.
            2. Calculate an output Tensor for every synapse storing a one if a synapases end is connected
               to a cell which is active for the given timeStep or a zero if not.
                    This will involve looking at the colIndex and CellIndex stored in the distalSynapse tensor
                    and then using these as indicies in the activeCellsTime tensor and comparing the
                    two timeSteps at each location to the current Timestep.
            3. Use the tensors in 1. and 2. and see which locations both contain a 1. This means that 
               distal synpase has a permannece larger then the connected permanence value and it's
               end is connected to a cell that is active for the given timestep. 
            4. The connected and active synapses from 3. should be counted up for each segment.
               The output should be a 3D tensor with the count number for each segment. This is the 
               predictionLevel for each segment.
            5. Update the activeSegsTime tensor by seeing if each segments count from step 4. is larger 
               then the self.activationThreshold.
            6. For all the segments that are active find the most active in a column. Store the activity count
               the segment and cell index for this column and whether it was predicting. The following is required for
               each column;
                    predictionLevel, mostPredSegmentIndex, mostPredCellIndex, columnPredicting 
            7. Set the most predicting cell in the column as the predicting cell.
            8. Create a new update structure if the cell wasn't already predicting
            9. Update the segment s by adding to the update tensors. The update happens in the future.
            10. Return the self.predictCellsTime storing the time steps when each cell was last in the predicting state.
            ''' 

        # Take a number of slices from the distal synapse tensor to seperate the column indicies,
        # cell indices and synapses permanences from the tensor and use them in the following functions.
        sliced_distalSyn_colInd = distalSynapses[:,:,:,:,0]
        sliced_distalSyn_cellInd = distalSynapses[:,:,:,:,1]
        sliced_distalSyn_perm = distalSynapses[:,:,:,:,2]
        self.segConActiveSynCount = self.connectActSynCount(sliced_distalSyn_colInd,
                                                   sliced_distalSyn_cellInd, 
                                                   sliced_distalSyn_perm,
                                                   timeStep, activeCellsTime,
                                                   self.connectPermanence)
        #print "connectActSynCount = \n%s" % self.segConActiveSynCount

        # Update the activeSegsTime tensor by seeing if each segments count is larger then the self.activationThreshold.
        self.activeSegsTime = self.updateActSegTimes(self.segConActiveSynCount,
                                                     self.activeSegsTime,
                                                     timeStep, 
                                                     self.activationThreshold)

        #print "self.activeSegsTime = \n%s" % self.activeSegsTime
        # For all the segments that are active find the most active in a column. Store the activity count
        # the segment and cell index for this column and whether it was predicting. 
        # self.mostPredSegInfo returns:
        #   [[mostPredSegmentInd array],
        #    [mostPredCellInd array], 
        #    [columnPredicting array]] 
        # The predictionLevel for each column is already known from the self.segConActiveSynCount tensor.
        mostActSegInfo = self.mostPredSegInfo(self.segConActiveSynCount,
                                              self.activationThreshold,
                                              self.numColumns)
        #print "mostActSegInfo = \n%s" % mostActSegInfo

        # Non theano functions. The code below does not uses any theano functions.
        # Update the self.predictCellsTime storing the last two times when a cell was predicting.
        # Create the update structure so a cells distal synapse permanences can be altered in the future.
        for c in range(self.numColumns):
            # Get from the mostActSegInfo tensor whether the current column is predicting or not.
            columnPredicting = mostActSegInfo[2][c]
            if columnPredicting == 1:
                mostPredCell = mostActSegInfo[1][c]
                # Set the most predicting cell in the column as the predicting cell.
                self.setPredictCell(c, mostPredCell, timeStep)
                # Only create a new update structure if the cell wasn't already predicting
                if self.checkCellPredicting(c, mostPredCell, timeStep-1) is False:
                    # Update the segment s by adding to the update tensors. The update happens in the future.
                    mostPredSegment = mostActSegInfo[0][c]
                    self.segIndUpdate[c][mostPredCell] = mostPredSegment
                    self.segActiveSyn[c][mostPredCell] = self.getSegmentActiveSynapses(distalSynapses[c][mostPredCell][mostPredSegment],
                                                                                       timeStep,
                                                                                       activeCellsTime)
                    # print "self.segActiveSyn[%s][%s] = %s" % (c, mostPredCell, self.segActiveSyn[c][mostPredCell])
        # print "self.currentSegSynCount = \n%s" % self.currentSegSynCount
        #print "self.predictCellsTime = \n%s" % self.predictCellsTime
        return self.predictCellsTime



def updateActiveCells(numColumns, cellsPerColumn, timeStep):
    # Update the tensor representing the last two times each cell was active.
    # Set a random selective to active at the current timeStep

    activeCells = np.random.randint(timeStep+1, size=(numColumns, cellsPerColumn, 2))
    # print "activeColumns = \n%s" % activeColumns
    return activeCells

if __name__ == '__main__':
    # A main function to test and debug this class.
    numRows = 1
    numCols = 3
    cellsPerColumn = 4
    numColumns = numRows * numCols
    maxSegPerCell = 3
    maxSynPerSeg = 3
    connectPermanence = 0.3
    activationThreshold = 1
    timeStep = 1

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

    # Create the active cells
    activeCells = np.zeros((numColumns, cellsPerColumn, 2))

    activeSeg = np.zeros((numColumns, cellsPerColumn, maxSegPerCell))

    # print "activeCells = \n%s" % activeCells
    print "distalSynapses = \n%s" % distalSynapses

    predCellsCalc = predictCellsCalculator(numColumns,
                                           cellsPerColumn,
                                           maxSegPerCell,
                                           maxSynPerSeg,
                                           connectPermanence,
                                           activationThreshold)
    # Run through calculator
    test_iterations = 1
    for i in range(test_iterations):
        timeStep += 1
        if timeStep % 20 == 0:
            print timeStep
        print "timeStep = \n%s" % timeStep
        # Change the active columns and active cells and run again.
        activeCells = updateActiveCells(numColumns, cellsPerColumn, timeStep)
        print "activeCells = \n%s" % activeCells
        predictCellsTime = predCellsCalc.updatePredictiveState(timeStep, activeCells, distalSynapses)
        print "predictCellsTime = \n%s" % predictCellsTime
        segIndUpdate, segActiveSyn = predCellsCalc.getSegUpdates()
        print "segIndUpdate = \n%s" % (segIndUpdate)
        print "segActiveSyn = \n%s" % (segActiveSyn)
        activeSegsTime = predCellsCalc.getActiveSegTimes()
        print "activeSegsTime = \n%s" % (activeSegsTime)



