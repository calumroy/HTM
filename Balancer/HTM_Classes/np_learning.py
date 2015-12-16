
import numpy as np
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

        # Store the previous colPotInputs.
        # This is so a potential synapse can work out if it's end
        # has changed state. If so then we update the synapses permanence.
        # Initialize with a negative value so the first update always updates
        # the permanence values. Normally this matrix holds 0 or 1 only.
        self.prevColPotInputs = np.array([[-1 for x in range(self.numPotSynapses)] for y in range(self.numColumns)])
        self.prevActiveCols = np.array([-1 for i in range(self.numColumns)])

    def updatePermanence(self, c, s, colPotInputs, colSynPerm):
        # Check if the input that this
        # synapses is connected to is active.
        if colPotInputs[c][s] == 1:
        # Only handles binary input sources
            colSynPerm[c][s] += self.spatialPermanenceInc
            colSynPerm[c][s] = min(1.0, colSynPerm[c][s])
        else:
            colSynPerm[c][s] -= self.spatialPermanenceDec
            colSynPerm[c][s] = max(0.0, colSynPerm[c][s])

    def updatePermanenceValues(self, colSynPerm, colPotInputs, activeCols):
        # The inputs colSynPerm and colPotInputs are matricies.
        # colSynPerm is the permanence values of every poetnetial synapse for each column.
        # colPotInputs is the binary input value for each potential synapse for each column.
        # If one then that potential synapse is connected to an active input bit.
        # activeCols is an array storing a bit indicating if the column is active (1) or not (0).
        for c in range(len(activeCols)):
            # Only update the potential synapses for the active columns.
            if activeCols[c] == 1:
                # If the column was newly activated then update all the permanence values
                # for each potential synapse regardless whether the synpases input has changed.
                if self.prevActiveCols[c] != activeCols[c]:
                    for s in range(len(colSynPerm[c])):
                        self.updatePermanence(c, s, colPotInputs, colSynPerm)
                else:
                    # The column was previously active.
                    # This means it it temporally pooling or the same input
                    # was sent. Only update the potential synpase permanences if
                    # the input to a potential synpase changed.
                    for s in range(len(colSynPerm[c])):
                        # Check that this potential synapses input has changed
                        if self.prevColPotInputs[c][s] != colPotInputs[c][s]:
                            self.updatePermanence(c, s, colPotInputs, colSynPerm)

        # Store the current inputs to the potentialSynapses to use next time.
        self.prevColPotInputs = colPotInputs
        self.prevActiveCols = activeCols

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


