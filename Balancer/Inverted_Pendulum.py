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

def createInput(angle, gridWidth, gridHeight, angleOverlap, minAngle, maxAngle):
    # Create the angle input matrix
    # angle = The current angle of the inverted pendulum
    # gridWidth = The width of the matrix
    # gridHeight = The height of the matrix
    # angleOverlap = The number of columns the active angle cells take up. This affects the overlap between angle readings.
    # minAngle = The minimum angle. This is the angle value that the first column represents.
    # maxAngle = The maximum angle. This is the angle value that the last column represents.
    if angle=='none':
        return np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
    angleInput = np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
    anglePos = float(float(gridWidth)/float(1.0+(abs(maxAngle-minAngle))))*float(abs(angle-minAngle))
    #print"anglePos,angleOverlap = %s,%s "%(anglePos,angleOverlap)
    #angleCol = int(round(anglePos))
    for row in range(len(angleInput)):
        for col in range(len(angleInput[0])):
            if col >= (round(anglePos-angleOverlap)) and col <= (round(anglePos+angleOverlap)):
                angleInput[row][col] = 1
    #print "grid = ",angleInput
    return angleInput

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
    def __init__(self):
        self.length = 1.0
        self.weigth = 1.0
        self.cartAcc = 1.0   # The carts acceleration in the horizontal direction m/s.
        self.x = 0.0
        self.y = self.length
        self.angle = 90      # 90 deg is upright
        self.vel = 0

    def step(self, acc, minAngle, maxAngle, maxAcc, time):
        # Calculate the new position of the pendulum after the time while applying the specified acceleration.
        # Simple pendulum for now!
        print " self.angle = %s, vel = %s, acc = %s, time = %s"%(self.angle,self.vel,acc,time)
        self.vel = self.vel+int(time*acc)
        # Limit the velocity
        if self.vel > maxAcc:
            self.vel = maxAcc
        if self.vel < -maxAcc:
            self.vel = -maxAcc
        self.angle = self.angle+self.vel
        # Limit the angle
        if self.angle > maxAngle:
            self.angle = maxAngle
        if self.angle < minAngle:
            self.angle = minAngle
        return self.angle


