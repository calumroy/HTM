from HTM_calc import np_inhibition
import numpy as np
import math
import random
import theano.tensor as T
from theano import function
from theano import scan
from HTM_calc import theano_predictCells as tpc

class test_theano_predictCells:
    def setUp(self):
        '''
        

        '''

    def test_case1(self):
        '''
        Test the theano predict Cells calculator class.
        '''
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

        # Create the active cells defining the timestep when the cells where last active.
        # Each cell stores the last 2 timesteps when it was active.
        activeCells = np.array(
            [[[1, 2],
              [1, 1],
              [2, 0],
              [2, 0]],

             [[1, 2],
              [2, 2],
              [0, 0],
              [1, 2]],

             [[1, 2],
              [2, 2],
              [1, 2],
              [2, 1]]])

        activeSeg = np.zeros((numColumns, cellsPerColumn, maxSegPerCell))

        # print "activeCells = \n%s" % activeCells
        print "distalSynapses = \n%s" % distalSynapses



        predCellsCalc = tpc.predictCellsCalculator(numColumns,
                                                   cellsPerColumn,
                                                   maxSegPerCell,
                                                   maxSynPerSeg,
                                                   connectPermanence,
                                                   activationThreshold)

        segConActiveSynCount = np.array([[[2,2,2],
                                          [2,1,2],
                                          [2,1,3],
                                          [2,2,2]],

                                         [[2,1,2],
                                          [2,2,2],
                                          [2,3,3],
                                          [2,1,3]],

                                         [[2,1,2],
                                          [2,3,3],
                                          [3,2,1],
                                          [3,1,2]]]
                                        )

        #import ipdb; ipdb.set_trace()
        # Run through calculator
        test_iterations = 1
        for i in range(test_iterations):
            timeStep += 1
            if timeStep % 20 == 0:
                print timeStep
            print "timeStep = \n%s" % timeStep
            # Change the active columns and active cells and run again.
            #activeCells = updateActiveCells(numColumns, cellsPerColumn, timeStep)
            print "activeCells = \n%s" % activeCells
            predictCellsTime = predCellsCalc.updatePredictiveState(timeStep, activeCells, distalSynapses)
            print "predictCellsTime = \n%s" % predictCellsTime

            segConActiveSynCount = predCellsCalc.getSegConActiveSynCount()
            print "segConActiveSynCount = \n%s" % segConActiveSynCount
            segIndUpdate, segActiveSyn = predCellsCalc.getSegUpdates()
            print "segIndUpdate = \n%s" % (segIndUpdate)
            print "segActiveSyn = \n%s" % (segActiveSyn)
            activeSegsTime = predCellsCalc.getActiveSegTimes()
            print "activeSegsTime = \n%s" % (activeSegsTime)

        # Expected Results
        ex_connectActSynCount = np.array(
        [[[1, 1, 2],
          [2, 2, 2],
          [2, 1, 3],
          [2, 2, 2]],

         [[3, 1, 3],
          [1, 2, 0],
          [2, 3, 2],
          [1, 2, 2]],

         [[2, 0, 2],
          [0, 3, 3],
          [2, 2, 2],
          [2, 2, 2]]])
        ex_predictCellsTime = np.array(
        [[[-1, -1],
          [-1, -1],
          [ 2, -1],
          [-1, -1]],

         [[ 2, -1],
          [-1, -1],
          [-1, -1],
          [-1, -1]],

         [[-1, -1],
          [ 2, -1],
          [-1, -1],
          [-1, -1]]])
        ex_segIndUpdate = np.array(
        [[-1, -1,  2, -1],
         [ 0, -1, -1, -1],
         [-1,  1, -1, -1]])
        ex_segActiveSyn = np.array(
        [[[-1, -1, -1],
          [-1, -1, -1],
          [ 1,  1,  1],
          [-1, -1, -1]],

         [[ 1,  1,  1],
          [-1, -1, -1],
          [-1, -1, -1],
          [-1, -1, -1]],

         [[-1, -1, -1],
          [ 1,  1,  1],
          [-1, -1, -1],
          [-1, -1, -1]]])
        ex_activeSegsTime = np.array(
        [[[0.,  0.,  2.],
          [2.,  2.,  2.],
          [2.,  0.,  2.],
          [2.,  2.,  2.]],

         [[2.,  0.,  2.],
          [0.,  2.,  0.],
          [2.,  2.,  2.],
          [0.,  2.,  2.]],

         [[2.,  0.,  2.],
          [0.,  2.,  2.],
          [2.,  2.,  2.],
          [2.,  2.,  2.]]])

        # Make sure the uninhibted columns are equal to the above
        # predetermined test results.
        assert np.array_equal(ex_predictCellsTime, predictCellsTime)

        assert np.array_equal(ex_segIndUpdate, segIndUpdate)

        assert np.array_equal(ex_segActiveSyn, segActiveSyn)

        assert np.array_equal(ex_activeSegsTime, activeSegsTime)