import numpy as np
from copy import deepcopy


class measureTemporalPooling:
    '''
    The purpose of this class is to measure the amount of temporal pooling
    occuring across a set of input grids. This means measure the amount that
    the input grids change by.

    This class stores the input grid it receives.
    It then uses this to compare to future grid arrays.
    It creates a running average of how much each successive
    grid changes from the previous one.
    '''
    def __init__(self):
        self.grid = None
        # A running average totalling the percent of temporal pooling.
        self.temporalAverage = 0
        self.numInputGrids = 0

    def temporalPoolingPercent(self, grid):
        if self.grid is not None:
            totalPrevActiveIns = np.sum(self.grid != 0)
            totalAndGrids = np.sum(np.logical_and(grid != 0, self.grid != 0))
            if totalPrevActiveIns > 0:
                percentTemp = float(totalAndGrids) / float(totalPrevActiveIns)
            else:
                # In this case there is no active columns. This means the spatial
                # pooler is not working or the inputs are empty!
                percentTemp = 0
            #print "         totalAndGrids = %s" % totalAndGrids
            self.temporalAverage = (float(percentTemp) +
                                    float(self.temporalAverage*(self.numInputGrids-1)))/float(self.numInputGrids)
            #print "         percentTemp = %s" % percentTemp
        self.grid = deepcopy(grid)
        self.numInputGrids += 1
        return self.temporalAverage
