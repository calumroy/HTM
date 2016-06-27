import numpy as np
import math

'''
A class to calculate the temporal pooling for a HTM layer.


'''


class TemporalPoolCalculator():
    def __init__(self, numColumns, numPotSynapses, spatialPermanenceInc, spatialPermanenceDec):
        self.numColumns = numColumns
        self.numPotSynapses = numPotSynapses
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec

        # Store the previous colPotInputs.
        # This is so a potential synapse can work out if it's end
        # has changed state. If so then we update the synapses permanence.
        # Initialize with a negative value so the first update always updates
        # the permanence values. Normally this matrix holds 0 or 1 only.
        self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevColActive = np.array([-1 for i in range(self.numColumns)])

        # Save the calculated vector describing if each column
        # was active but not bursting one timestep ago.
        self.colActNotBurstVect = None

    def updateProximalTempPool(self, colPotInputs,
                               colActive, colPotSynPerm, timeStep):
        '''
        Update the proximal synapses (the column synapses) such that;
            a. For each currently active column increment the permenance values
               of potential synapses connected to an active input one timestep ago.
            b. For each column that was active one timestep ago increment the permenance
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
            # Update the potential synapses for the currently active columns.
            if colActive[c] == 1:
                for s in range(len(colPotSynPerm[c])):
                    # If any of the columns potential synapses where connected to an
                    # active input increment the synapses permenence.
                    if self.prevColPotInputs[c][s] == 1:
                        print "Current active Col prev input active for col, syn = %s, %s" % (c, s)
                        colPotSynPerm[c][s] += self.spatialPermanenceInc
                        colPotSynPerm[c][s] = min(1.0, colPotSynPerm[c][s])
            # Update the potential synapses for the previous active columns.
            if self.prevColActive[c] == 1:
                # If any of the columns potential synapses are connected to a
                # currently active input increment the synapses permenence.
                if colPotInputs[c][s] == 1:
                    print "Prev active Col current input active for col, syn = %s, %s" % (c, s)
                    colPotSynPerm[c][s] += self.spatialPermanenceInc
                    colPotSynPerm[c][s] = min(1.0, colPotSynPerm[c][s])

        # Store the current inputs to the potentialSynapses to use next time.
        self.prevColPotInputs = colPotInputs
        self.prevColActive = colActive

        return colPotSynPerm


if __name__ == '__main__':

    numRows = 4
    numCols = 4
    spatialPermanenceInc = 1.0
    spatialPermanenceDec = 0.2
    maxNumTempoPoolPatterns = 3
    activeColPermanenceDec = float(spatialPermanenceInc)/float(maxNumTempoPoolPatterns)
    numPotSyn = 4
    timeStep = 1
    numColumns = numRows * numCols
    # Create an array representing the permanences of colums synapses
    colPotSynPerm = np.random.rand(numColumns, numPotSyn)
    # Create an array representing the potential inputs to each column
    colPotInputs = np.random.randint(2, size=(numColumns, numPotSyn))
    # Create an array representing the active columns
    colActive = np.random.randint(2, size=(numColumns))

    print "INITIAL colPotSynPerm = \n%s" % colPotSynPerm
    print "colPotInputs = \n%s" % colPotInputs
    print "colActive = \n%s" % colActive

    tempPooler = TemporalPoolCalculator(numColumns, numPotSyn, spatialPermanenceInc, spatialPermanenceDec)

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
