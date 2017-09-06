#!/usr/bin/python

"""
This code simulates an inverted pendulum balancing on a moving stand.
The HTM controls the motion of the stand and hence this affects the pendulum.

author: Calum Meiklejohn
website: calumroy.com
"""
import numpy as np
import math
import random



def medianAcc(accGrid, minAcc, maxAcc):
    # Take a command grid input and output an average acceleration value
    # The minimum acceleration is the acceleration represented by column 0 the
    # maximum acceleration is from the most right column; linearly scaled between this.
    numPredCells = 0 # This is the number of predicting cells
    numAccLevels = 1+abs(maxAcc-minAcc) # The number of accleration levels
    gridWidth = len(accGrid[0])
    accArray = np.array([0 for i in range(numAccLevels)])
    # The acceleration represented per column. Plus one since -1 to 1 is 3 columns as 0 must be represented.
    accPerCol = float((1+abs(maxAcc-minAcc))/float(gridWidth))
    for k in range(len(accGrid)):
        for m in range(len(accGrid[k])):
            if accGrid[k][m]==1:
                #avgAcc += m*accPerCol
                accArray[int(m*accPerCol)] += 1;
                numPredCells += 1
    if numPredCells>0:  # Avoid divide by zero
        #avgAcc = int(avgAcc/numPredCells) + minAcc
        mostPredAcc = 0
        medianAcc = 0
        for i in range(len(accArray)):
            if accArray[i] > mostPredAcc:
                mostPredAcc = accArray[i]
                medianAcc = i
        medianAcc = medianAcc + minAcc
        #print "     Median Acceleration = %s accPerCol = %s numPredCells = %s"%(medianAcc,accPerCol,numPredCells)
        return medianAcc
    else:
        return 'none'




class InvertedPendulum():
    def __init__(self, gridWidth, gridHeight):
        self.length = 1.0
        self.weight = 1.0
        self.cartAcc = 1.0   # The carts acceleration in the horizontal direction m/s.
        self.x = 0.0
        self.y = self.length
        self.angle = int(gridWidth/2)      # 90 deg is upright
        self.vel = 1

        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.angleOverlap = 1
        self.minAngle = -round(gridWidth/2)
        self.maxAngle = round(gridWidth/2)
        self.minAcc = -1
        self.maxAcc = 1

    def step(self, cellGrid):
        # Convert the input grid input an acceleration first
        acc = self.convertSDRtoAcc(cellGrid)
        # Calculate the new position of the pendulum after applying the specified acceleration.
        # Simple pendulum for now!

        self.vel = acc
        self.angle = self.angle + self.vel
        # Limit the velocity
        if self.vel > self.maxAcc:
            self.vel = self.maxAcc
        if self.vel < -self.maxAcc:
            self.vel = -self.maxAcc
        self.angle = self.angle+self.vel
        # Limit the angle and set velocity to zero
        if self.angle > self.maxAngle:
            self.angle = self.maxAngle
            self.vel = 0
        if self.angle < self.minAngle:
            self.angle = self.minAngle
            self.vel = 0
        print " self.angle = %s, vel = %s, acc = %s " % (self.angle, self.vel, acc)
        return self.angle

    def getReward(self):
        # Look at the current state of the simulation and decide if a reward should be given
        reward = 0
        if self.angle >= -5 and self.angle <= 5:
            reward = 1
        return reward

    def createSimGrid(self):
        # Create the angle input matrix
        # angle = The current angle of the inverted pendulum
        # gridWidth = The width of the matrix
        # gridHeight = The height of the matrix
        # angleOverlap = The number of columns the active angle cells take up. This affects the overlap between angle readings.
        # minAngle = The minimum angle. This is the angle value that the first column represents.
        # maxAngle = The maximum angle. This is the angle value that the last column represents.
        # return np.array([[0 for i in range(self.gridWidth)] for j in range(self.gridHeight)])
        angleInput = np.array([[0 for i in range(self.gridWidth)] for j in range(self.gridHeight)])
        anglePos = float(float(self.gridWidth)/float(1.0+(abs(self.maxAngle-self.minAngle))))*float(abs(self.angle-self.minAngle))
        #print"anglePos,angleOverlap = %s,%s "%(self.anglePos,self.angleOverlap)
        #angleCol = int(round(self.anglePos))
        for row in range(len(angleInput)):
            for col in range(len(angleInput[0])):
                if col >= (round(anglePos-self.angleOverlap)) and col <= (round(anglePos+self.angleOverlap)):
                    angleInput[row][col] = 1
        #print "grid = ",angleInput
        return angleInput

    def convertSDRtoAcc(self, cellGrid):
        # Convert a sparse distributed representation into an acceleration
        # Each cell output represents a particular acceleration command. In
        # this simple case we are using -1, 0 or 1 m/s^2. Future mapping techniques
        # should just use a random mapping value. The total average of the output
        # accelerations is calculated and returned.

        # The idea is that the HTM will learn about this mapping and eventually choose
        # the right cells so the output from the HTM commands the
        # "correct" acceleration to control the system.

        acceleration = 0
        height = len(cellGrid)
        width = len(cellGrid[0])
        accRange = abs(self.minAcc - self.maxAcc)
        numActiveCells = 0

        for row in range(height):
            for col in range(width):
                if cellGrid[row][col] == 1:
                    # Not a random mapping but close enough. We aren't using completely random
                    # since we want the same mapping each time.
                    #accCell = ((col + row) % accRange) + self.minAcc
                    accCell = float(col)/float(width) * float(accRange) + float(self.minAcc)
                    acceleration += accCell
                    # Calculate the total number of active cells
                    numActiveCells += 1
        # Prevent divide by zero!
        if numActiveCells != 0:
            acceleration = float(acceleration)/float(numActiveCells)
        print "Num of active cells from command = %s" % numActiveCells
        print "Acceleration Command = %s" % acceleration
        return acceleration


class InputCreator:

    def __init__(self, gridWidth, gridHeight, overlap):
        self.pos_x = int(gridWidth/2)     # The column number that the vertical line is at.
        self.direction = 1  # ! if the vertical line is moving right -1 if it's moving left.
        self.overlap = overlap  # The number of columns that the line can overlap either side.
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

    def newInput(self):
        # If the vertical line is at column zero or at the end column then reverse the direction.
        if self.pos_x >= self.gridWidth-1 and self.direction == 1:
            self.direction = -1
        if self.pos_x <= 0 and self.direction == -1:
            self.direction = 1
        # Move the vertical lines postion
        self.pos_x = self.pos_x + self.direction

        return self.createInput(self.pos_x, self.gridWidth, self.gridHeight, self.overlap)

    def createInput(self, pos_x, gridWidth, gridHeight, overlap):
        newInput = np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
        for row in range(gridHeight):
            for col in range(gridWidth):
                if col >= (round(pos_x-overlap)) and col <= (round(pos_x+overlap)):
                    #Add some noise if you feel like it
                    if random.randint(0, 5) >= 0:
                        newInput[row][col] = 1
        #print "grid = ",newInput
        return newInput
