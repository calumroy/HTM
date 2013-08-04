#!/usr/bin/python

"""
HTM line pattern inputs

This code creates line patterns that can be fed into the HTM network.

author: Calum Meiklejohn
website: calumroy.com
last edited: August 2013
"""
import numpy as np
import math
  
def inputPatterns():
    # Temporary code to create pattern inputs
    pattern1=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    pattern2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]])
    pattern3=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0]])
    patternArray=[pattern1,pattern2,pattern3]
    return patternArray

def createPatternArray(arrayLength,gridWidth,gridHeight,lineWidth,lineYInt,lineAngle):
    c=lineYInt
    patternArray=np.array([createLinePattern(gridWidth,gridHeight,lineWidth,c,lineAngle)])
    print patternArray
    for i in range(arrayLength-1):  # Minus one since we already added a pattern to the array
        c-=2
        patternArray=np.vstack((patternArray,[createLinePattern(gridWidth,gridHeight,lineWidth,c,lineAngle)]))
    print patternArray
    return patternArray

   
def createLinePattern(gridWidth,gridHeight,lineWidth,c,angle):
    # Creates an input grid that has a line pattern at a certain polar angle in degrees.
    # transX and TransY are the translational x and y offset from the center of the 
    # grid.
    inputGrid=np.array([[0 for i in range(gridWidth)] for j in range(gridHeight)])
    gradient=math.sin(math.radians(angle))/math.cos(math.radians(angle))
    for x in range(gridWidth):
        for y in range(gridHeight):
            distance=abs(y-gradient*x+c)/math.sqrt(math.pow(gradient,2)+1)
            if distance<(lineWidth/2):
                #print (y,x)
                inputGrid[y][x]=1
    #print (gridWidth,gridHeight)
    #print inputGrid
    return inputGrid
            


