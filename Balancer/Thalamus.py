

"""
HTM GUI
author: Calum Meiklejohn
website: calumroy.com

This class is a simple thalamus class to be used by the HTM network.
The purpose of this class is to direct the HTM network to control the
outputs such that desired input states are reached.

"""
import numpy as np
import random
from utilities import sdrFunctions as SDRFunct
from operator import itemgetter


class Thalamus:
    def __init__(self, columnArrayWidth, columnArrayHeight):
        '''
        The thalamus contains a Qvalues grid variable whose purpose is to
        store in a 2d array Qvalues which are used and updated.
        These decided the thalamus output which directs the HTM
        to produce desired inputs.

        Qvalues must be larger then zero. If a square does not have
        a Qvalue then it will be zero. A squares Qvalue is updated
        when it is part of a command that received a reward or lead
        to a new Q state that has a non zero Qvalue (see Qlearning).
        '''
        self.width = columnArrayWidth
        self.height = columnArrayHeight
        self.QValues = np.array([[0.0 for i in range(self.width)]
                                for j in range(self.height)])

        # Policy Parameters
        # The number of squares that should be active for a new command.
        self.numSquaresInComm = 6
        # The chance a new untested square will be
        # tried out and returned as part of the new command.
        self.newCommChance = 0.05
        # A parameter specifing how importnat maximizing future Qvalues is.
        self.futureQvalMax = 0.9
        # Learning rate. This determines how quickly Qvalues change.
        self.qValLearnRate = 0.1

        # Store the last Qvalues which makes it into the last output.
        # This list is used to update those Qvalues, stored as (x,y,Qvalue).
        self.lastQValList = []
        # Store the last reward received. This is used to update Qvalues.
        self.reward = 0

    def rewardThalamus(self, reward):
        # A function that rewards the thalamus for doing something correct.
        # Note the reward is stored and used to update Qvalues after a
        # new command is chosen.
        if reward > 0:
            print "Reward recieved by thalamus= %s" % reward
        self.reward = reward

    def updateQvalues(self):
        '''
        Update the Qvalues depending on the output from the last command.
        The Qvalues contributing to the last command are stored in the
        variable self.lastQValList.

        The QValues are increased if they resulted in a new Qstate that
        has a higher Qvalue or a reward was received. A new Q state with a
        lower Qvalue will decrease the Qvalues.

        Qvalues are updated with the following formula
        sample = Reward + futureQvalMax * (nextAction maximizing Qvalues)
        QValue(x,y) = (1-qValLearnRate) * QValue(x,y) + qValLearnRate * sample

        '''
        # Get the last Qvalues that contributed to the command
        qValueTotal = 0.0
        for i in range(len(self.lastQValList)):
            # The list stores (x,y,Qvalue)
            qValueTotal += self.lastQValList[i][2]
        # Get the average of the last qVals
        qValAverage = float(qValueTotal)/(len(self.lastQValList))

        sample = self.reward + self.futureQvalMax * qValAverage
        if sample > 0:
            print "Sample = %s" % sample

        # Update each of the qvalues form the last command
        print "self.UpdatedQValues = "
        for i in range(len(self.lastQValList)):
            # The list stores (x,y,Qvalue)
            pos_x = self.lastQValList[i][0]
            pos_y = self.lastQValList[i][1]
            self.QValues[pos_y][pos_x] = ((1 - self.qValLearnRate) * self.QValues[pos_y][pos_x] +
                                          self.qValLearnRate * sample)
            print "\n %s " % self.QValues[pos_y][pos_x]

    def pickCommand(self, predCommand):
        '''
        From the predicting cells from the input select those
        with the highest Q values.

        The input is a grid which represents cells that are predicting.
        These cells corresponding Qvalues will be used to select the
        output command.

        The Qvalues are updated after this new command is chosen.
        This needs to be done now to use the reward from the
        previous Q state.
        '''
        nominatedQvalGrid = self.getNomQvalues(predCommand)

        # From the nominated qvalues choose the highest values.
        # If no values or not enough are nominated use a policy to
        # Decide which Qvalues shall make up the output command.
        newCommand = self.commandPolicy(nominatedQvalGrid)

        # Update the QValues using this new command.
        # This needs to be done now to use the reward from the
        # previous Q state
        self.updateQvalues()

        return newCommand

    def commandPolicy(self, nominatedQvalGrid):
        # Using the policy and the input nominated QValues return a new
        # binary grid representing the chosen output command grid.
        width = len(self.QValues[0])
        height = len(self.QValues)
        newCommand = SDRFunct.returnBlankSDRGrid(width, height)
        highestQValList = self.getHighestQvalues(nominatedQvalGrid)

        print "HighestQVal list = \n %s" % highestQValList
        # Store in a list (x,y,QValue).
        self.lastQValList = []
        # The number of random selections made
        numRandomChoices = 0
        # With a chance select each Qvalue to be part of the new command.
        for i in range(len(highestQValList)):
            currentQval = highestQValList[i][2]
            # If the Qvalue is very high then there is less chance of
            # picking a new Q value. Normalise it with the learning rate.
            if random.random() >= self.newCommChance*(self.qValLearnRate)/(currentQval):
                pos_x = highestQValList[i][0]
                pos_y = highestQValList[i][1]
                self.lastQValList.append([pos_x, pos_y, highestQValList[i][2]])
            else:
                numRandomChoices += 1

        # If the return list of positions and Qvalues is too small
        # then add a random selection of squares to fill up the command.
        if len(self.lastQValList) < self.numSquaresInComm:
            print " Thalamus choosing %s random outputs" % (self.numSquaresInComm - len(self.lastQValList))
            for i in range(self.numSquaresInComm - len(self.lastQValList)):
                # Choose a random Qvalue to add to the new command
                pos_x = random.randint(0, width - 1)
                # TODO
                # Fix the awkward y scaling of the output command.
                # Times the pos_y by two since the command input
                # is half the total input to the HTM.
                pos_y = random.randint(0, int((height - 1)/2))
                self.lastQValList.append([pos_x, pos_y, self.QValues[pos_y][pos_x]])

        # Set the squares to true in the new command
        for i in range(len(self.lastQValList)):
            pos_x = self.lastQValList[i][0]
            pos_y = self.lastQValList[i][1]
            # TODO
            # Fix the awkward y scaling of the output command.
            # Times the pos_y by two since the command input
            # is half the total input to the HTM.
            newCommand[pos_y*2][pos_x] = 1
        return newCommand

    def getHighestQvalues(self, nominatedQvalGrid):
        # Return a list of the highest Qvalues. Return the number
        # specified by the parameter self.numSquaresInComm
        # Store the position of the highest Qvalues in a list
        # Each position in the list will store (x,y,Qvalue)
        qValList = []
        width = len(nominatedQvalGrid[0])
        height = len(nominatedQvalGrid)
        for y in range(height):
                for x in range(width):
                    currQval = nominatedQvalGrid[y][x]
                    if currQval > 0:
                        qValList.append([x, y, currQval])

        # Sort the qVal List by the Qvalue
        qValList = sorted(qValList, key=itemgetter(2))
        # Only the self.numSquaresInComm number of Qvalues
        # are needed. If not enough Qvalue exist in the list return
        # a smaller or empty list.
        qValList = qValList[:self.numSquaresInComm]
        #print " Thalamus highest QvalList = %s" % qValList
        return qValList

    def getNomQvalues(self, grid):
        # If a square in the input grid is active then return a
        # grid containing the Qvalue for that square else return 0.
        # If the input grid is smaller then the Qvalue grid then
        # just use the top most part.

        if (len(grid) != len(self.QValues)) or (len(grid[0]) != len(self.QValues[0])):
            print "WARNING Thalamus QValue Grid is not the same size as grid input!"
            #SDRFunct.joinInputArrays()
        if (len(grid) <= len(self.QValues)) and (len(grid[0]) <= len(self.QValues[0])):
            # Remember and return the number of nominated Q values
            numberNomQVals = 0
            width = len(grid[0])
            height = len(grid)
            nominatedQValGrid = np.array([[0.0 for i in range(width)] for j in range(height)])
            for y in range(height):
                for x in range(width):
                    if grid[y][x] == 1:
                        nominatedQValGrid[y][x] = self.QValues[y][x]
                        numberNomQVals += 1
            print "numberNomQVals = %s " % (numberNomQVals)
            return nominatedQValGrid

        print " ERROR Thalamus grid input is larger then the QValues grid!"
        return None

