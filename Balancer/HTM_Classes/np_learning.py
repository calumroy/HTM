import theano.tensor as T
from theano import function
import numpy as np
from theano.sandbox.neighbours import images2neibs
from theano import Mode
import math


'''
A class used to increase or decrease the permanence
values of the potential synapses in a single HTM layer.
This class uses normal numpy arrays and is a CPU implementation.

This class requires as inputs:
    * The current permanence values for each cols potential synapse.
    * A list of the connected and unconnected potential synpases for
        each column.
    * A list of the active columns.
    * How much to increment or decrement synapse values.

THIS THEANO LEARNING CLASS IS A REIMPLEMENTATION OF THE ORIGINAL CODE:
    for c in self.activeColumns:
        for s in c.potentialSynapses:
            # Check if the input that this
            #synapses is connected to is active.
            inputActive = self.Input[s.pos_y][s.pos_x]
            if inputActive == 1:
            #Only handles binary input sources
                s.permanence += c.spatialPermanenceInc
                s.permanence = min(1.0, s.permanence)
            else:
                s.permanence -= c.spatialPermanenceDec
                s.permanence = max(0.0, s.permanence)

    for i in range(len(self.columns)):
        for c in self.columns[i]:
            c.minDutyCycle = 0.01*self.maxDutyCycle(self.neighbours(c))
            c.updateBoost()

            if c.overlapDutyCycle < c.minDutyCycle:
                self.increasePermanence(c, 0.1*self.connectPermanence)

'''


class LearningCalculator():
    def __init__(self,
                 numColumns,
                 numPotSynapses,
                 spatialPermanenceInc,
                 spatialPermanenceDec):
        self.numColumns = numColumns
        self.numPotSynapses = numPotSynapses
        self.spatialPermanenceInc = spatialPermanenceInc
        self.spatialPermanenceDec = spatialPermanenceDec

    def updatePermanenceValues(self, colSynPerm, colPotInputs, activeCols):
        # The inputs colSynPerm and colPotInputs are matricies.
        # colSynPerm is the permanence values of every poetnetial synapse for each column.
        # colPotInputs is the input value for each potential synapse for each column.
        # If one then that potential synapse is connected to an active input bit.
        # activeCols is an array storing a bit indicating if the column is active (1) or not (0).
        for c in range(len(activeCols)):
            if activeCols[c] == 1:
                for s in range(len(colSynPerm[c])):

                    # Check if the input that this
                    #synapses is connected to is active.
                    if colPotInputs[c][s] == 1:
                    #Only handles binary input sources
                        colSynPerm[c][s] += self.spatialPermanenceInc
                        colSynPerm[c][s] = min(1.0, colSynPerm[c][s])
                    else:
                        colSynPerm[c][s] -= self.spatialPermanenceDec
                        colSynPerm[c][s] = max(0.0, colSynPerm[c][s])

        return colSynPerm


if __name__ == '__main__':

    numRows = 4
    numCols = 4
    spatialPermanenceInc = 1.0
    spatialPermanenceDec = 1.0
    numPotSyn = 4
    numColumns = numRows * numCols
    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # Create an array representing the potential inputs to each column
    colPotInputsMat = np.random.randint(2, size=(numColumns, numPotSyn))
    # Create an array representing the active columns
    activeCols = np.random.randint(2, size=(numColumns))

    print "colSynPerm = \n%s" % colSynPerm
    print "colPotInputsMat = \n%s" % colPotInputsMat
    print "activeCols = \n%s" % activeCols

    permanenceUpdater = LearningCalculator(numColumns,
                                           numPotSyn,
                                           spatialPermanenceInc,
                                           spatialPermanenceDec)

    colSynPerm = permanenceUpdater.updatePermanenceValues(colSynPerm,
                                                          colPotInputsMat,
                                                          activeCols)


