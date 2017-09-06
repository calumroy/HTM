import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math

'''
A class to calculate the temporal pooling for a HTM layer.
This class uses theano functions to speed up the computation.
It can be implemented on a GPU see theano documents for
enabling GPU calculations.

Inputs:
It uses the overlap values for each column and an matrix
of values specifying when a column was active but not bursting last.

Outputs:
It outputs a matrix of new overlap values for each column where
the columns that are temporally pooling are given a maximum overlap value.


THIS TEMPORAL THEANO CLASS IS A REIMPLEMENTATION OF THIS CODE:
    # Temporal pooling is done if the column was active but not bursting one timestep ago.
    if self.columnActiveNotBursting(c, self.timeStep-1) is not None:

        if c.overlap >= c.minOverlap:
            # The col has a good overlap value and should allow temp pooling
            # to continue on the next time step. Set the time flag to not the
            # current time to allow this (we'll use zero).
            c.stopTempAtTime = 0

        # If the time flag for temporal pooling was not set to now
        # then we should perform temporal pooling.
        if c.stopTempAtTime != (self.timeStep):
            if c.overlap < c.minOverlap:
                # The current col has a poor overlap and should stop temporal
                # pooling on the next timestep.
                c.stopTempAtTime = self.timeStep+1
            # Recalculate the overlap using the potential synapses not just the connected.
            c.overlap = 0
            for s in c.potentialSynapses:
                # Check if the input that this synapses is connected to is active.
                inputActive = self.Input[s.pos_y][s.pos_x]
                c.overlap = c.overlap + inputActive
            # If more potential synapses then the min overlap
            # are active then set the overlap to the maximum value possible.
            if c.overlap >= c.minOverlap:
                maxOverlap = (c.potentialWidth)*(c.potentialHeight)
                c.overlap = c.overlap + maxOverlap
                c.lastTempPoolingTime = self.timeStep
'''


class TemporalPoolCalculator():
    def __init__(self, potentialWidth, potentialHeight,
                 minOverlap):
        # Temporal Parameters
        ###########################################
        self.potentialWidth = potentialWidth
        self.potentialHeight = potentialHeight
        self.minOverlap = minOverlap
        self.maxOverlap = potentialWidth * potentialHeight

        # Save the calculated vector describing if each column
        # was active but not bursting one timestep ago.
        self.colActNotBurstVect = None

        # Create theano variables and functions
        ############################################

        # Create the theano function for calculating
        # if the column was active but not bursting one timestep ago.
        # Outputs a vector where a one represents a col active not burst at timestep.
        self.colActNotBurst = T.vector(dtype='float32')
        self.timeStepVect1 = T.vector(dtype='float32')
        self.moreRecent = T.switch(T.eq(self.colActNotBurst, self.timeStepVect1),
                                   1.0, 0.0)
        self.m = self.moreRecent
        self.doTempPool = function([self.colActNotBurst, self.timeStepVect1],
                                   self.m,
                                   mode=Mode(linker='vm'),
                                   allow_input_downcast=True)

        # Create the theano function for calculating
        # the time to stop temporal pooling for each column.
        self.j_stopAfter = T.vector('stopTempAfterTime', dtype='float32')
        self.l_timeStep = T.vector(dtype='float32')
        self.k_doTemp = T.vector(dtype='float32')
        self.overlapVal = T.vector(dtype='float32')
        # If the matrix value is more then zero then the
        # The col has a good overlap value and should allow temp pooling
        # to continue on the next time step. Set the time flag to not the
        # current time to allow this (we'll use zero).
        # Set the stop time to zero if overlap is larger then zero. This allows
        # temporal pooling to continue on the next time step since overlap was good.
        # If the stopTemppooling time is not equal to the current timestep and
        # the overlap is equal to zero then set the stop temp pool time to the next timestep.
        self.checkOverlapZero = T.switch(T.gt(self.overlapVal, 0.0), 0.0, self.j_stopAfter)
        self.checkOverlap = T.switch(T.eq(self.overlapVal, 0.0), self.l_timeStep+1, self.j_stopAfter)
        self.setTimeToNext = T.switch(T.eq(self.j_stopAfter, self.l_timeStep), self.checkOverlapZero, self.checkOverlap)
        self.checkTimeToNext = T.switch(T.gt(self.k_doTemp, 0.0), self.setTimeToNext, self.l_timeStep)
        # Use enable downcast so the numpy arrays of float 64 can be downcast to float32
        self.setStopTempPoolTimeZero = function([self.overlapVal,
                                                self.j_stopAfter,
                                                self.l_timeStep,
                                                self.k_doTemp],
                                                self.checkTimeToNext,
                                                mode=Mode(linker='vm'),
                                                allow_input_downcast=True)

        # Create the theano function for calculating
        # if the stopTempPool time flag is not set to now
        # then the overlap should be recalculated using the potential synpases.
        # If this new overlap value is larger then minOverlap then set it to
        # overlap plus maxOverlap value.
        self.stopTempTime = T.vector(dtype='float32')
        self.timeStepVect2 = T.vector(dtype='float32')
        self.potSynInputs = T.matrix(dtype='float32')
        self.oldOverlapVal = T.vector(dtype='float32')
        self.m_potSum = self.potSynInputs.sum(axis=1)
        self.checkMinOverlap = T.switch(T.lt(self.m_potSum, self.minOverlap), 0.0, self.oldOverlapVal + self.maxOverlap)
        self.checkPotOverlap = T.switch(T.eq(self.stopTempTime, self.timeStepVect2), self.oldOverlapVal, self.checkMinOverlap)
        # Use enable downcast so the numpy arrays of float 64 can be downcast to float32
        self.calcPotOverlap = function([self.stopTempTime,
                                        self.timeStepVect2,
                                        self.potSynInputs,
                                        self.oldOverlapVal],
                                       self.checkPotOverlap,
                                       allow_input_downcast=True)

    def getColActNotBurstVect(self):
        # Return the binary vector displaying if a column was active
        # but not bursting one timestep ago.
        # This should only be called after calculateTemporalPool is run.
        return self.colActNotBurstVect

    def calculateTemporalPool(self, colActNotBurst, timeStep, colOverlapVals,
                              colInputPotSyn, colStopTempAtTime):
        # First check if the column should perform temp pooling.
        # This is done for all columns that where active but not bursting.
        # Need to create a timestep matrix with same dimension as colActNotBurst.
        # print "colActNotBurst = \n%s" % colActNotBurst
        #print "timeStep = %s" % timeStep
        numCols = len(colActNotBurst)
        # Setup a matrix to compare the previous time to.
        prevTimeStepVect = np.array([timeStep - 1 for j in range(numCols)])
        # Setup a vector to compare the current time to.
        timeStepVect = np.array([timeStep for j in range(numCols)])
        # print "prevTimeStepVect = \n%s" % prevTimeStepVect
        self.colActNotBurstVect = self.doTempPool(colActNotBurst, prevTimeStepVect)
        # In the self.colActNotBurstVect each pos represents a col. If it has non zero
        # value then that col should do temp pooling (the column was active but not bursting
        # one timestep ago).
        # Update the columns stopTempAtTime variable.
        # If the overlap is less then minOverlap (zero since the
        # overlap values have already filtered smaller values to zero),
        #
        # print "colOverlapVals = \n%s" % colOverlapVals
        # print "colStopTempAtTime = \n%s" % colStopTempAtTime
        # print "timeStepVect = \n%s" % timeStepVect
        # print "self.colActNotBurstVect = \n%s" % self.colActNotBurstVect
        updatedTempStopTime = self.setStopTempPoolTimeZero(colOverlapVals,
                                                           colStopTempAtTime,
                                                           timeStepVect,
                                                           self.colActNotBurstVect
                                                           )
        # print "updatedTempStopTime = \n%s" % updatedTempStopTime
        # print "timeStepVect = \n%s" % timeStepVect
        # print "doTempPoolVect = \n%s" % doTempPoolVect
        # print "colInputPotSyn = \n%s" % colInputPotSyn
        # print "colOverlapVals = \n%s" % colOverlapVals
        newTempPoolOverlapVals = self.calcPotOverlap(updatedTempStopTime,
                                                     timeStepVect,
                                                     colInputPotSyn,
                                                     colOverlapVals
                                                     )
        # print "newTempPoolOverlapVals = \n%s" % newTempPoolOverlapVals

        return newTempPoolOverlapVals, updatedTempStopTime


if __name__ == '__main__':

    potWidth = 2
    potHeight = 2
    centerPotSynapses = 1
    minOverlap = 2
    numCols = 16
    timeStep = 4

    tempPooler = TemporalPoolCalculator(potWidth, potHeight, minOverlap)

    # Some made up inputs to test with
    colActNotBurst = np.random.randint(7, size=numCols)
    colOverlapVals = np.random.randint(potWidth * potHeight, size=(numCols))
    colInputPotSyn = np.random.randint(2, size=(numCols, potWidth * potHeight))
    colStopTempAtTime = np.random.randint(2, size=(numCols))
    # To get the above input array from a htm use something like the following
    # allCols = self.htm.regionArray[0].layerArray[0].columns.flatten()
    # colActNotBurst = np.array([allCols[j].activeStateArray for j in range(1600)])
    # colStopTempAtTime = np.array([allCols[j].stopTempAtTime for j in range(1600)])
    # Get colInputPotSyn from the overlap theano class.

    newTempPoolOverlapVals, updatedTempStopTime = tempPooler.calculateTemporalPool(colActNotBurst, timeStep, colOverlapVals,
                                                                                   colInputPotSyn, colStopTempAtTime)

