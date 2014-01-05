#!/usr/bin/python

"""
This code simulates an inverted pendulum balancing on a moving stand.
The HTM controls the motion of the stand and hence this affects the pendulum.

author: Calum Meiklejohn
website: calumroy.com
last edited: August 2013
"""
import numpy as np
import math
import random

def createInput(angle, gridWidth, gridHeight, angleOverlap, minAcc, maxAcc):
    # Create the angle input matrix
    # angle = The current angle of the inverted pendulum
    # gridWidth = The width of the matrix
    # gridHeight = The height of the matrix
    # angleOverlap = The number of columns the active angle cells take up. This affects the overlap between angle readings.
    # minAcc = The minimum acceleration. This is the acceleration value that the first column represents. 
    # maxAcc = The maximum acceleration. This is the acceleration value that the last column represents. 
    if angle=='none':
        return np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
    angleInput = np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
    anglePos = float(gridWidth)/float(abs(minAcc)+abs(maxAcc))*float(abs(angle-minAcc))
    #print"anglePos = %s, angleOverlap = %s"%(anglePos,angleOverlap)
    #angleCol = int(round(anglePos))
    for row in range(len(angleInput)):
        for col in range(len(angleInput[0])):
            if col > (anglePos-angleOverlap) and col < (anglePos+angleOverlap): 
                angleInput[row][col] = 1
    #print "grid = ",angleInput
    return angleInput

def averageAcc(accGrid):
    # Take a command grid input and output an average acceleration value
    avgAcc = 0 
    numPredCells = 0 # This is the number of predicting cells
    numberCols = len(accGrid[0])
    for k in range(len(accGrid)):
        for m in range(len(accGrid[k])):
            avgAcc += k
            numPredCells += 1
    avgAcc = round(avgAcc/numPredCells)
    # Convert the column number into an acceleration. Left is neg right is pos
    avgAcc = -round(numberCols/2)+avgAcc
    print "     Average Acceleration = %s"%avgAcc
    return avgAcc
        
        


class InvertedPendulum():
    def __init__(self):
        self.length = 1.0
        self.weigth = 1.0
        self.cartAcc = 1.0   # The carts acceleration in the horizontal direction m/s.
        self.x = 0.0
        self.y = self.length
        self.angle = 90      # 90 deg is upright
        self.vel = 0

    def step(self, acc, time):
        # Calculate the new position of the pendulum after the time while applying the specified acceleration.
        # Simple pendulum for now! ]
        print "self.angle = %s, vel = %s, time = %s, acc = %s"%(self.angle,self.vel,time,acc)
        self.vel = self.vel+int(time*acc/4)
        # Limit the velocity
        if self.vel > 10:
            self.vel = 10
        if self.vel < -10:
            self.vel = -10
        self.angle = self.angle+self.vel
        # Limit the angle
        if self.angle > 180:
            self.angle = 180
        if self.angle < 0:
            self.angle = 0
        return self.angle


