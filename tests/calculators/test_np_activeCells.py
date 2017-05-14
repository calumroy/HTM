from HTM_calc import np_inhibition
import numpy as np
import math
import random
import theano.tensor as T
from theano import function
from theano import scan
from HTM_calc import np_activeCells as nac

class test_theano_predictCells:
    def setUp(self):
        '''


        '''

    # Helper functions for the Main function.
    def updateActiveCols(self, numColumns):
        activeColumns = np.random.randint(2, size=(numColumns))
        print "activeColumns = \n%s" % activeColumns
        return activeColumns

    def test_case1(self):
        '''
        Test the numpy active Cells calculator class.
        '''
        numRows = 1
        numCols = 3
        cellsPerColumn = 4
        numColumns = numRows * numCols
        maxSegPerCell = 3
        maxSynPerSeg = 3
        connectPermanence = 0.3
        activationThreshold = 1
        minNumSynThreshold = 1
        minScoreThreshold = 1
        newSynPermanence = 0.3
        connectPermanence = 0.2
        timeStep = 1

        # Create the distalSynapse 5d tensor holding the information of the distal synapses.
        distalSynapses = np.array(
        [[[[[1., 1., 0.2],
            [0., 1., 0.3],
            [2., 2., 0.7]],

           [[2., 2., 0.1],
            [2., 1., 0.8],
            [2., 0., 0.2]],

           [[0., 3., 0.3],
            [1., 2., 0.8],
            [1., 1., 0.4]]],


          [[[2., 2., 0.5],
            [2., 0., 0.1],
            [0., 2., 0.4]],

           [[1., 1., 0.9],
            [1., 3., 0.3],
            [1., 2., 0.4]],

           [[1., 2., 0.3],
            [1., 1., 1. ],
            [0., 2., 0.7]]],


          [[[0., 3., 0.3],
            [0., 1., 0.8],
            [2., 3., 0.7]],

           [[1., 1., 0.7],
            [2., 2., 0.2],
            [0., 1., 0.6]],

           [[0., 2., 0.5],
            [1., 0., 1. ],
            [0., 2., 0.9]]],


          [[[1., 0., 0.6],
            [2., 1., 0.2],
            [1., 3., 0.7]],

           [[2., 3., 0.6],
            [0., 1., 0.9],
            [1., 0., 0.5]],

           [[0., 1., 0.9],
            [1., 3., 0.3],
            [2., 2., 1. ]]]],



         [[[[2., 0., 0.4],
            [0., 2., 0.7],
            [2., 2., 1. ]],

           [[2., 1., 0.1],
            [0., 2., 0.4],
            [1., 2., 0.1]],

           [[2., 3., 0.9],
            [1., 1., 0.3],
            [1., 1., 0.5]]],


          [[[1., 2., 0.9],
            [2., 2., 0.3],
            [1., 2., 0.8]],

           [[1., 1., 0.3],
            [0., 2., 1. ],
            [1., 2., 1. ]],

           [[1., 2., 0.8],
            [0., 1., 0. ],
            [2., 1., 0.2]]],


          [[[0., 1., 0. ],
            [2., 2., 0.3],
            [1., 3., 0.8]],

           [[0., 0., 0.3],
            [1., 0., 0.9],
            [2., 2., 0.9]],

           [[1., 3., 0.4],
            [0., 3., 0. ],
            [0., 3., 1. ]]],


          [[[1., 2., 1. ],
            [1., 2., 0.1],
            [2., 2., 0.9]],

           [[1., 2., 0.1],
            [1., 0., 0.5],
            [1., 1., 0.9]],

           [[0., 3., 0. ],
            [2., 2., 0.6],
            [1., 1., 0.5]]]],



         [[[[0., 3., 0.5],
            [2., 1., 0.5],
            [0., 1., 0.3]],

           [[0., 1., 0.3],
            [1., 2., 0.7],
            [0., 1., 0.6]],

           [[0., 3., 0.3],
            [2., 1., 0.7],
            [0., 0., 0.2]]],


          [[[0., 3., 0.1],
            [1., 3., 0. ],
            [0., 2., 0.1]],

           [[1., 3., 0.6],
            [2., 1., 0.4],
            [0., 2., 0.6]],

           [[1., 3., 1. ],
            [2., 2., 0.5],
            [2., 3., 0.7]]],


          [[[0., 3., 0. ],
            [1., 0., 0.8],
            [1., 1., 0.9]],

           [[0., 2., 1. ],
            [2., 2., 0.8],
            [0., 1., 0.3]],

           [[1., 2., 0.2],
            [1., 0., 0.6],
            [1., 3., 0.7]]],


          [[[2., 0., 0.1],
            [2., 1., 0.7],
            [2., 2., 0.9]],

           [[2., 2., 0. ],
            [2., 3., 0.4],
            [1., 0., 0.6]],

           [[2., 1., 0.5],
            [2., 2., 0.1],
            [2., 2., 0.5]]]]]
        )

        # Create the predictive cells defining the timestep when the cells where last predicting.
        # Each cell stores the last 2 timesteps when it was predicting.
        predictCells = np.array(
            [[[1, 0],
              [0, 0],
              [0, 0],
              [0, 0]],

             [[1, 0],
              [1, 2],
              [1, 0],
              [1, 0]],

             [[1, 1],
              [2, 2],
              [1, 2],
              [2, 1]]])

        activeSeg = np.array(
            [[[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[1, 0, 0],
              [1, 2, 0],
              [1, 0, 0],
              [1, 0, 0]],

             [[1, 1, 0],
              [2, 2, 0],
              [1, 2, 0],
              [2, 1, 0]]])
        # Set the active columns
        activeColumns = np.array([1, 0, 0])

        # print "activeCells = \n%s" % activeCells
        print "distalSynapses = \n%s" % distalSynapses
        print "predictCells = \n%s" % predictCells

        actCellsCalc = nac.activeCellsCalculator(numColumns,
                                                 cellsPerColumn,
                                                 maxSegPerCell,
                                                 maxSynPerSeg,
                                                 minNumSynThreshold,
                                                 minScoreThreshold,
                                                 newSynPermanence,
                                                 connectPermanence)

        #import ipdb; ipdb.set_trace()
        # Run through calculator
        test_iterations = 2
        for i in range(test_iterations):
            timeStep += 1
            if timeStep % 20 == 0:
                print timeStep
            print "timeStep = \n%s" % timeStep
            activeCells, learnCells = actCellsCalc.updateActiveCells(timeStep,
                                                                     activeColumns,
                                                                     predictCells,
                                                                     activeSeg,
                                                                     distalSynapses)

            print "activeCells = \n%s" % activeCells
            print "learnCells = \n%s" % learnCells
            # Change the active columns and active cells and run again.
            activeColumns = np.array([1, 1, 0])
        # Expected Results
        ex_activeCells = np.array(
                        [[[ 2,  3],
                          [-1, -1],
                          [-1, -1],
                          [-1, -1]],

                         [[-1, -1],
                          [ 3, -1],
                          [-1, -1],
                          [-1, -1]],

                         [[-1, -1],
                          [-1, -1],
                          [-1, -1],
                          [-1, -1]]])
        ex_learnCells = np.array(
                        [[[ 2,  3],
                          [-1, -1],
                          [-1, -1],
                          [-1, -1]],

                         [[-1, -1],
                          [ 3, -1],
                          [-1, -1],
                          [-1, -1]],

                         [[-1, -1],
                          [-1, -1],
                          [-1, -1],
                          [-1, -1]]])

        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(ex_activeCells, activeCells)
        assert np.array_equal(ex_learnCells, learnCells)

