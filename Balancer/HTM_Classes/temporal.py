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

        # Create theano variables and functions
        ############################################

        # Create the theano function for calculating
        # if the column was active but not bursting one timestep ago.
        # Outputs a matrix where a one represents col active not burst at timestep.
        self.colActNotBurst = T.matrix(dtype='float32')
        self.timeStepMat = T.matrix(dtype='float32')
        self.moreRecent = T.switch(T.eq(self.colActNotBurst, self.timeStepMat),
                                   1.0, 0.0)
        self.m = self.moreRecent.sum(axis=1)
        self.doTempPool = function([self.colActNotBurst, self.timeStepMat],
                                   self.m,
                                   mode=Mode(linker='vm'),
                                   allow_input_downcast=True)

        # Create the theano function for calculating
        # if the overlap is larger then zero and therefore temporal
        # pooling should continue.
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
        # the overlap is larger then zero set the stop temp pool time to the next timestep.
        self.checkOverlapZero = T.switch(T.gt(self.overlapVal, 0.0), 0.0, self.j_stopAfter)
        self.checkOverlap = T.switch(T.eq(self.overlapVal, 0.0), self.l_timeStep+1, self.j_stopAfter)
        self.setTimeToNext = T.switch(T.eq(self.j_stopAfter, self.l_timeStep), self.checkOverlapZero, self.checkOverlap)
        self.checkTimeToNext = T.switch(T.gt(self.k_doTemp, 0.0), self.setTimeToNext, self.j_stopAfter)
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
        self.n_doTemp = T.vector(dtype='float32')
        self.potSynInputs = T.matrix(dtype='float32')
        self.oldOverlapVal = T.vector(dtype='float32')
        self.m_potSum = self.potSynInputs.sum(axis=1)
        self.checkMinOverlap = T.switch(T.lt(self.m_potSum, self.minOverlap), 0.0, self.oldOverlapVal + self.maxOverlap)
        self.checkPotOverlap = T.switch(T.gt(self.n_doTemp, 0.0), self.checkMinOverlap, self.oldOverlapVal)
        # Use enable downcast so the numpy arrays of float 64 can be downcast to float32
        self.calcPotOverlap = function([self.n_doTemp,
                                       self.potSynInputs,
                                       self.oldOverlapVal],
                                       self.checkPotOverlap,
                                       allow_input_downcast=True)

    def calculateTemporalPool(self, colActNotBurst, timeStep, colOverlapVals,
                              colInputPotSyn, colStopTempAtTime):
        # First check if the column should perform temp pooling.
        # This is done for all columns that where active but not bursting.
        # Need to create a timestep matrix with same dimensions as colActNotBurst.
        print "colActNotBurst = \n%s" % colActNotBurst
        numCols = len(colActNotBurst)
        historyLen = len(colActNotBurst[0])
        timeStepMat = np.array([[timeStep for i in range(historyLen)] for j in range(numCols)])
        timeStepVect = np.array([timeStep for j in range(numCols)])
        print "timeStepMat = \n%s" % timeStepMat
        doTempPoolMat = self.doTempPool(colActNotBurst, timeStepMat)
        print "doTempPoolMat = \n%s" % doTempPoolMat
        # In the doTempPoolMat each pos represents a col. If it has non zero
        # value then that col should do temp pooling.
        # Update the columns stopTempAtTime variable.
        # If the overlap is less then minOverlap (zero since the
        # overlap values have already filtered smaller values to zero).
        print "colOverlapVals = \n%s" % colOverlapVals
        print "colStopTempAtTime = \n%s" % colStopTempAtTime
        print "timeStepVect = \n%s" % timeStepVect
        updatedTempStopTime = self.setStopTempPoolTimeZero(colOverlapVals,
                                                           colStopTempAtTime,
                                                           timeStepVect,
                                                           doTempPoolMat
                                                           )
        print "updatedTempStopTime = \n%s" % updatedTempStopTime
        print "doTempPoolMat = \n%s" % doTempPoolMat
        print "colInputPotSyn = \n%s" % colInputPotSyn
        print "colOverlapVals = \n%s" % colOverlapVals
        newTempPoolOverlapVals = self.calcPotOverlap(doTempPoolMat,
                                                     colInputPotSyn,
                                                     colOverlapVals
                                                     )
        print "newTempPoolOverlapVals = \n%s" % newTempPoolOverlapVals


if __name__ == '__main__':

    potWidth = 2
    potHeight = 2
    centerPotSynapses = 1
    minOverlap = 2
    historyLen = 2
    numCols = 16
    timeStep = 4

    tempPooler = TemporalPoolCalculator(potWidth, potHeight, minOverlap)

    # Some made up inputs to test with
    colActNotBurst = np.random.randint(7, size=(numCols, historyLen))
    colOverlapVals = np.random.randint(potWidth * potHeight, size=(numCols))
    colInputPotSyn = np.random.randint(2, size=(numCols, potWidth * potHeight))
    colStopTempAtTime = np.random.randint(2, size=(numCols))
    # To get the above input array from a htm use something like the following
    # allCols = self.htm.regionArray[0].layerArray[0].columns.flatten()
    # colActNotBurst = np.array([allCols[j].activeStateArray for j in range(1600)])

    tempPooler.calculateTemporalPool(colActNotBurst, timeStep, colOverlapVals,
                                     colInputPotSyn, colStopTempAtTime)

