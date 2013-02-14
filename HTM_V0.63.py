# Title: HTM
# Description: git managed development of a HTM network
# Author: Calum Meiklejohn
# Development phase: alpha V 0.63

import HTM_draw
#import pygame
import numpy as np
import random
import math
import pprint

Struct = {'field1': 'some val', 'field2': 'some val'}
myStruct = { 'num': 1}


class Synapse:
    def __init__(self,input,pos_x,pos_y):
            self.pos_x=pos_x            # The start of the synapse the other end is at a columns position 
            self.pos_y=pos_y
            self.sourceInput=input[pos_y][pos_x]
            self.permanence = 0.3
            self.source_input_index = 0.0
            #If the permanence value for a synapse is greater than this
            #value, it is said to be connected.
            self.connectPermanence = 0.2
    def updateInput(self,input):
        self.sourceInput=input[self.pos_y][self.pos_x]

class Segment:
    def __init__(self,length):
        self.predict = False
        self.index = -1
        #self.activeSynapses = np.array(Synapse(1)) #NEED TO FINISH THIS WITH THE SYNAPSE SOURCE INPUT
        self.sequenceSegment = False

class Cell:
    def __init__(self):
        length = 10
        # dendrite segments
        segments = np.array(Segment(length))
        # State of the cell
        active = False
        predict = False   

class Column:
    def __init__(self, length, pos_x,pos_y,input):
        self.cells = np.array(Cell())
        for i in range(length):
            self.cells = np.hstack((self.cells,[Cell()]))
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.overlap = 0.0
        self.minOverlap = 4
        self.boost = 1
        self.inhibitionRadius = 2   # The max distance a column can inhibit another column
        self.potentialRadius = 1    # The max distance that Synapses can be made at
        self.permanenceInc = 0.1
        self.permanenceDec = 0.02
        self.minDutyCycle = 0.01   # The minimum firing rate of the column
        self.activeDutyCycleArray = np.array([0.0]) # Keeps track of when the column was active. All columns start as active. It stores the numInhibition time when the column was active
        self.activeDutyCycle = 0.0 # the firing rate of the column
        self.activeState = False
        self.overlapDutyCycle = 0.0 # The rate at which the overlap is larger then the min overlap
        self.overlapDutyCycleArray = np.array([0.0]) # Keeps track of when the colums overlap was larger then the minoverlap
        self.boostStep = 0.1
        self.connectedSynapses=np.array([],dtype=object)
        self.potentialSynapses=np.array([],dtype=object) # the possible feed forward Synapse connections for the column
        #Work out the potential feedforward connections this column could make
        for i in range(int(self.pos_y-self.potentialRadius),int(self.pos_y+self.potentialRadius)+1):
            if i>=0 and i<(len(input)):
                for j in range(int(self.pos_x-self.potentialRadius),int(self.pos_x+self.potentialRadius)+1):
                    if j>=0 and j<(len(input[0])):
                        self.potentialSynapses=np.append(self.potentialSynapses,[Synapse(input,j,i)])   #i is pos_y j is pos_x
    def updateConnectedSynapses(self):
        self.connectedSynapses=np.array([],dtype=object)
        for i in range(len(self.potentialSynapses)):
            if self.potentialSynapses[i].permanence>self.potentialSynapses[i].connectPermanence:
                self.connectedSynapses = np.append(self.connectedSynapses,self.potentialSynapses[i])
    def input(self,input):
        for i in range(len(self.potentialSynapses)):
            self.potentialSynapses[i].updateInput(input)
    def updateBoost(self):
        if self.activeDutyCycle<self.minDutyCycle:
            self.boost = self.boost+self.boostStep
        else:
            #print "activeDutyCycle %s > minDutyCycle %s" %(self.activeDutyCycle,self.minDutyCycle)
            self.boost = 1.0
        #print self.boost
class HTMLayer:
    def __init__(self, input, column_array_width,column_array_height):
        
        # The cell array is a 1Dimensional array of cell columns
        # They are in a 2 Dimensiuonal array column_array_width by column_array_height.
        # This might be a crap idea
        self.width = column_array_width
        self.height = column_array_height
        self.input = input
        self.desiredLocalActivity = 3
        self.cellsPerColumn = 3
        self.activationThreshold = 3
        self.connectPermanence = 0.2
        self.learningRadius = 4
        self.initialPerm = 0.3
        #This is also defined in the Synapse class!!! Maybe change this
        self.connectedPerm = 0.2    # The value a connected Synapses must be higher then.
        self.minThreshold = 10
        self.newSynapseCount = 5
        self.dutyCycleAverageLength = 20
        self.timeStep = 0                     
        self.output = np.array([[0 for i in range(self.width)] for j in range(self.height)])
        self.activeColumns = np.array([],dtype=object)
        self.averageReceptiveFeildSizeArray = np.array([])
        self.columns = np.array([[Column(self.cellsPerColumn,i,j,input) for i in range(column_array_width)]
        for j in range(column_array_height)],dtype=object)      #Create the array storing the columns
    def updateOutput(self):
        for i in range(len(self.output)):
            for j in range(len(self.output[i])):
                self.output[i][j] = 0
        for i in range(len(self.activeColumns)):
            x = self.activeColumns[i].pos_x
            y = self.activeColumns[i].pos_y
            self.output[y][x] = 1
    def potentialSynapses(c):
        pass
    def neighbours(self,c):
        close_columns=np.array([],dtype=object)     # returns a list of the columns that are within the inhibitionRadius of c
        for i in range(int(c.pos_y-c.inhibitionRadius),int(c.pos_y+c.inhibitionRadius)+1):
            if i>=0 and i<(len(input)):
                for j in range(int(c.pos_x-c.inhibitionRadius),int(c.pos_x+c.inhibitionRadius)+1):
                    if j>=0 and j<(len(input[0])):
                        close_columns = np.append(close_columns,self.columns[i][j])
        return close_columns
    def updateOverlapDutyCycle(self,c):
            c.overlapDutyCycleArray = np.append(c.overlapDutyCycleArray,self.timeStep)   # Append the current time to the list of times that the column was active for
            for i in range(len(c.overlapDutyCycleArray)):        # Remove the values that where too long ago
                if c.overlapDutyCycleArray[0]<(self.timeStep-self.dutyCycleAverageLength):
                    c.overlapDutyCycleArray=np.delete(c.overlapDutyCycleArray,0,0)
                else:
                   break
            c.overlapDutyCycle = float(len(c.overlapDutyCycleArray))/float(self.dutyCycleAverageLength)     #Update the overlap duty cycle running average
            #print "overlap DutyCycle = %s length = %s averagelength = %s"%(c.overlapDutyCycle,len(c.overlapDutyCycleArray),self.dutyCycleAverageLength)
    
    def increasePermanence(self,c,scale):
        for i in range(len(c.potentialSynapses)):
            c.potentialSynapses[i].permanence = (1+scale)*(c.potentialSynapses[i].permanence)  # Increase the permance by a scale factor
    def boostFunction(c):
        pass
    def kthScore(self,cols,kth):
        if len(cols)>0 and kth>0 and kth<(len(cols)-1):
            orderedScore = np.array(cols[0].overlap)
            #print cols[0].overlap
            for i in range(1,len(cols)):    #Add the overlap values to a single list
                orderedScore = np.append(orderedScore,[cols[i].overlap])
            orderedScore=np.sort(orderedScore)
            #print orderedScore
            return orderedScore[-kth]       # Minus since list starts at lowest   
        return 0
    def averageReceptiveFeildSize(self):
        self.averageReceptiveFeildSizeArray = np.array([])
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                self.averageReceptiveFeildSizeArray = np.append(self.averageReceptiveFeildSizeArray,len(c.connectedSynapses))
        print np.average(self.averageReceptiveFeildSizeArray)
        return int(math.sqrt(np.average(self.averageReceptiveFeildSizeArray))/2)    #Returns the radius of the average receptive feild size
    def updateActiveDutyCycle(self,c):
        if c.activeState==True:
            c.activeDutyCycleArray = np.append(c.activeDutyCycleArray,self.timeStep)   # Append the current time call to the list of times that the column was active for
        for i in range(len(c.activeDutyCycleArray)):        # Remove the values that where too long ago
            if c.activeDutyCycleArray[0]<(self.timeStep-self.dutyCycleAverageLength):
                c.activeDutyCycleArray=np.delete(c.activeDutyCycleArray,0,0)
            else:
                break
        c.activeDutyCycle = float(len(c.activeDutyCycleArray))/float(self.dutyCycleAverageLength)     #Update the active duty cycle running average
        #print "DutyCycle = %s length = %s averagelength = %s"%(c.activeDutyCycle,len(c.activeDutyCycleArray),self.dutyCycleAverageLength)
    def maxDutyCycle(self,cols):
        maxActiveDutyCycle = 0.0        
        for c in cols:
            if maxActiveDutyCycle<c.activeDutyCycle:
                maxActiveDutyCycle = c.activeDutyCycle
        return maxActiveDutyCycle
    def Cell(c,i):
        pass
    def activeColumns(t):
        pass
    def activeState(c,i,t):
        pass
    def predictiveState(c,i,t):
        pass
    def learnState(c,i,t):
        pass
    def segmentActive(s,t,state):
        pass
    def getActiveSegment(s,t,state):
        pass
    def getBestMatchingSegment(c,i,t):
        pass    
    def getBestMatchingCell(c):
        pass
    def getSegmentActiveSynapses(c,i,t,s,newSynapses=False):
        pass
    def adaptSegment(segmentList,positiveReinforcement):
        pass   
    def Input(self,Input):
        self.input = Input
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.input(Input)
    def Overlap(self):
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.overlap = 0.0
                c.updateConnectedSynapses()
                for s in c.connectedSynapses:
                    c.overlap = c.overlap + s.sourceInput
                if c.overlap<c.minOverlap:
                    c.overlap=0.0
                else:
                    c.overlap=c.overlap*c.boost
                    self.updateOverlapDutyCycle(c)
                #print "%d %d %d" %(c.overlap,c.minOverlap,c.boost)
    def inhibition(self):
        self.activeColumns=np.array([],dtype=object)
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                minLocalActivity = self.kthScore(self.neighbours(c),self.desiredLocalActivity)  
                #print "current column = (%s,%s)"%(c.pos_x,c.pos_y)
                if c.overlap>0 and c.overlap>=minLocalActivity:
                    self.activeColumns=np.append(self.activeColumns,c)
                    c.activeState = True
                    #print "x,y = %s,%s overlap = %d min = %d" %(c.pos_x,c.pos_y,c.overlap,minLocalActivity)
                else:
                    c.activeState = False
                self.updateActiveDutyCycle(c)       # Update the active duty cycle variable of every column
    def learning(self): # NOT FINISHED
        for c in self.activeColumns:
            for s in c.potentialSynapses:
                if s.sourceInput==1: #Only handles binary input sources
                    s.permanence += c.permanenceInc
                    s.permanence = min(1.0,s.permanence)
                else:
                    s.permanence -= c.permanenceDec
                    s.permanence = max(0.0,s.permanence)
        average = self.averageReceptiveFeildSize() #Find the average of the receptive feild sizes just once
        #print "inhibition radius = %s" %average
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                c.minDutyCycle = 0.01*self.maxDutyCycle(self.neighbours(c))
                c.updateBoost()
                c.inhibitionRadius = average # Set to the average of the receptive feild sizes. All columns have the same inhibition radius
                if c.overlapDutyCycle<c.minDutyCycle:
                    self.increasePermanence(c,0.1*self.connectPermanence)
        self.updateOutput()
                
class HTM:
    def __init__(self, numLayers,input, column_array_width,column_array_height):
        # The class contains multiple HTM layers stack on one another
        self.numberLayers = numLayers
        self.width = column_array_width
        self.height = column_array_height
        self.HTMLayerArray = np.array([],dtype = object)
        for i in range(numLayers):
            self.HTMLayerArray = np.append(self.HTMLayerArray,HTMLayer(input,self.width,self.height))
    def learn(self):
        for i in range(len(self.HTMLayerArray)):
            self.HTMLayerArray[i].timeStep = self.HTMLayerArray[i].timeStep+1
            if i == 0:
                self.HTMLayerArray[i].Input(input)
            else:
                output = self.HTMLayerArray[i-1].output
                self.HTMLayerArray[i].Input(output)
            self.HTMLayerArray[i].Overlap()
            self.HTMLayerArray[i].inhibition()
            self.HTMLayerArray[i].learning()


def run_loop(HTM,input):
    HTM_draw.initialize_drawing()
    even = 0
    for j in range(100):
        even += 1
        print "NEW learning stage\n"
        # Added some noise with an alternating pattern for testing
        for k in range(len(input)):
            for l in range(len(input[k])):
                some_number = round(random.uniform(0,10))
                if some_number>8:
                    input[k][l] = 1
                else:
                    input[k][l] = 0
##        input = np.array([[round(random.uniform(0,1)) for i in
##         range(width)] for j in range(height)])
        if even % 2 == 0:
            print "EVEN"
            input[2][3:7] = 1
            input[3][7] = 1 
            input[4][7] = 1 
            input[5][7] = 1 
            input[6][3:7] = 1
            input[3][3] = 1 
            input[4][3] = 1 
            input[5][3] = 1 
        else:
            print "ODD"
            input[8][7:9] = 1
            input[7][7:9] = 1
            
        #Learning and updating
        HTM.learn()
        HTM_draw.draw_HTM(HTM,input)
    HTM_draw.quit_drawing()

if __name__ == "__main__":
    sizew = 15
    sizeh = 10
    input = np.array([[round(random.uniform(0,1)) for i in range(sizew)] for j in range(sizeh)])
    HTMNetwork = HTM(9,input,sizew,sizeh)
    run_loop(HTMNetwork,input)