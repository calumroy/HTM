# Title: HTM
# Description: git managed development of a HTM network
# Author: Calum Meiklejohn
# Development phase: alpha V 0.63

#import HTM_draw
#import pygame
import numpy as np
import random
import math
import pprint

##Struct = {'field1': 'some val', 'field2': 'some val'}
##myStruct = { 'num': 1}

SegmentUpdate = {'index' : '-1', 'activeSynapses' : '0', 'sequenceSegment' : 0 , 'createdAtTime' : 0}


class Synapse:
    def __init__(self,input,pos_x,pos_y,cellIndex):
            # cell is -1 if the synapse connects the HTM layers input.
            # Otherwise it is a horizontal connection to the cell 
            # index self.cell in the column at self.pos_x self.pos_y
            self.cell = cellIndex
            self.pos_x=pos_x            # The end of the synapse. The start is at a column or cells position
            self.pos_y=pos_y
            # sourceInput is 1 if the connected input is active 0 if not active and -1 if it's in predicted state.
            # The if the synapse is a vertical type then it can only have the vaules 0 or 1 since
            # it's connected to an input to the HTM layer and not a cell
            if cellIndex == -1: # If the created synapse is vertical connect to the input
                self.sourceInput=input[pos_y][pos_x]
            else:   # If the Synpase is horizontal then just set sourceInput to 0 it's unused 
                self.sourceInput=0  
            self.permanence = 0.31
            self.source_input_index = 0.0
            #If the permanence value for a synapse is greater than this
            #value, it is said to be connected.
            self.connectPermanence = 0.3
    def updateInput(self,input):
        # If Synapse is vertical then self.cell = -1 otherwise the synapse is horizontal
        # and connects to a cell within the HTM. If it is horizontal then there is no 
        # self.sourceInput
        if self.cell == -1:
            self.sourceInput=input[self.pos_y][self.pos_x]


class Segment:
    def __init__(self):
        self.predict = False
        self.index = -1
        self.sequenceSegment = 0    # Stores the last time step that this segment was predicting activity
        # Stores the synapses that have been created and have a larger permenence than 0.0
        self.synapses = np.array([],dtype = object) 


class Cell:
    def __init__(self):
        # dendrite segments
        self.numInitSegments = 1    # Must be greater then zero
        self.segments = np.array([],dtype=object)
        for i in range(self.numInitSegments):
            self.segments = np.hstack((self.segments,[Segment()]))
        # Create a dictionary to store the segmentUpdate structures
        self.segmentUpdateList = []
        # Segment update stucture holds the updates for the cell. These updates are made later.
        self.segmentUpdate = {'index':-1,'activeSynapses':np.array([],dtype=object),'newSynapses':np.array([],dtype=object),'sequenceSegment':0,'createdAtTime':0}
        for i in range(self.numInitSegments):
            self.segmentUpdateList.append(self.segmentUpdate.copy())
        #print self.segmentUpdateList
##        # State of the cell
##        self.active = False
##        self.predict = False   
    

class Column:
    def __init__(self, length, pos_x,pos_y,input):
        self.cells = np.array([],dtype=object)
        for i in range(length):
            self.cells = np.hstack((self.cells,[Cell()]))
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.overlap = 0.0
        self.minOverlap = 4
        self.boost = 1
        self.inhibitionRadius = 1   # The max distance a column can inhibit another column
        self.potentialRadius = 1    # The max distance that Synapses can be made at
        self.permanenceInc = 0.1
        self.permanenceDec = 0.05
        self.minDutyCycle = 0.01   # The minimum firing rate of the column
        self.activeDutyCycleArray = np.array([0]) # Keeps track of when the column was active. All columns start as active. It stores the numInhibition time when the column was active
        self.activeDutyCycle = 0.0 # the firing rate of the column
        self.activeState = False
        self.overlapDutyCycle = 0.0 # The rate at which the overlap is larger then the min overlap
        self.overlapDutyCycleArray = np.array([0]) # Keeps track of when the colums overlap was larger then the minoverlap
        self.boostStep = 0.1
        self.connectedSynapses=np.array([],dtype=object)
        self.potentialSynapses=np.array([],dtype=object) # the possible feed forward Synapse connections for the column
        #Work out the potential feedforward connections this column could make
        for i in range(int(self.pos_y-self.potentialRadius),int(self.pos_y+self.potentialRadius)+1):
            if i>=0 and i<(len(input)):
                for j in range(int(self.pos_x-self.potentialRadius),int(self.pos_x+self.potentialRadius)+1):
                    if j>=0 and j<(len(input[0])):
                        # Create a Synapse pointing to the HTM layers input so the synapse cellIndex is -1
                        self.potentialSynapses=np.append(self.potentialSynapses,[Synapse(input,j,i,-1)])   #i is pos_y j is pos_x
        self.predictiveStateArray = np.array([]) # An array storing when each of the cells in the column were last in a predictive state. 
        for i in range(length):
            self.predictiveStateArray = np.hstack((self.predictiveStateArray,[0]))
        # An array storing the timestep when each cell in the column was last in an 
        # active state. This means the column has 
        # feedforward input and the cell has a temporal context indicated by active segments.
        self.activeStateArray = np.array([]) 
        for i in range(length):
            self.activeStateArray = np.hstack((self.activeStateArray,[0]))
        self.learnStateArray = np.array([]) # An array storing when each of the cells in the column were last in a learn state. 
        for i in range(length):
            self.learnStateArray = np.hstack((self.learnStateArray,[0]))
    # POSSIBLY MOVE THESE FUNCTIONS TO THE HTMLayer CLASS?
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
    #def updateArray(self,timeStep,array):
        ## This function will be used if activeArray ends up storing more than just the last active time.
        # This function is used to update the predict, active and learn state arrays.
        # It adds the new time to the start of the list and deletes the last item in the list.
        # This way the newest times are always at the start of the list.
        #array.insert(0,timeStep)
        #del(array[len(array)-1])
        #return array
    
class HTMLayer:
    def __init__(self, input, column_array_width,column_array_height):
        
        # The columns are in a 2 dimensional array column_array_width by column_array_height.
        # This might be a crap idea
        self.width = column_array_width
        self.height = column_array_height
        self.input = input
        # The overlap values are used in determining the active columns. For columns with the same overlap value
        # both columns are active. This is why sometimes more columns then the desiredLocalActivity parameter
        # are observed in the inhibition radius.
        self.desiredLocalActivity = 1 # How many cells within the inhibition radius are active
        self.cellsPerColumn = 3
        self.connectPermanence = 0.3
        self.learningRadius = 4
        self.initialPerm = 0.3
        #This is also defined in the Synapse class!!! Maybe change this
        self.connectedPerm = 0.3    # The value a connected Synapses must be higher then.
        self.minThreshold = 3       # Should be smaller than activationThreshold
        self.newSynapseCount = 7    # This limits the activeSynapse array to this length. It should be renamed
        self.activationThreshold = 4    # How many synapses on a segment must be active for the segment to be active
        self.dutyCycleAverageLength = 1000
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
            if i>=0 and i<(len(self.columns)):
                for j in range(int(c.pos_x-c.inhibitionRadius),int(c.pos_x+c.inhibitionRadius)+1):
                    if j>=0 and j<(len(self.columns[0])):
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
        # Increases all the permanences of the Synapses.
        # It's used to help columns win that don't have a good overlap with any inputs
        for i in range(len(c.potentialSynapses)):
            c.potentialSynapses[i].permanence = min(1.0,(1+scale)*(c.potentialSynapses[i].permanence))  # Increase the permance by a scale factor
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
        #print np.average(self.averageReceptiveFeildSizeArray)
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
    def deleteWeakSynapses(self,c,i,segIndex):
        # Delete the synapses that have a permanence value that is too low form the segment.
        deleteActiveSynapses=[]  # This is a list of the indicies of the active synapses that will be deleted
        for k in range(len(c.cells[i].segments[segIndex].synapses)):
            syn = c.cells[i].segments[segIndex].synapses[k]
            if syn.permanence<syn.connectPermanence:
                deleteActiveSynapses.append(k)
        c.cells[i].segments[segIndex].synapses=np.delete(c.cells[i].segments[segIndex].synapses,deleteActiveSynapses)
        print "     deleted %s number of synapses"%(len(deleteActiveSynapses))
    def activeState(self,c,i,timeStep):
        if c.activeStateArray[i] == timeStep:
            return True
        else:
            return False 
    def predictiveState(self,c,i,timeStep):
        if c.predictiveStateArray[i] == timeStep:
            return True
        else:
            return False 
    def learnState(c,i,t):
        pass
    def randomActiveSynapses(self,c,i,s,timeStep):
        # Randomly add self.newSynapseCount-len(synapses) number of Synapses
        # that connect with cells that are active
        #print "randomActiveSynapses time = %s"%timeStep
        count = 0
        synapseList = np.array([],dtype=object) # A list of all potential synapses that are active
        for l in range(len(self.columns)):  # Can't use c since c already represents a column
            for m in self.columns[l]:       
                for j in range(len(m.learnStateArray)):
                    if  m.learnStateArray[j] == timeStep:
                        #print "time = %s synapse ends at active cell x,y,i = %s,%s,%s"%(timeStep,m.pos_x,m.pos_y,j)
                        synapseList = np.append(synapseList,Synapse(0,m.pos_x,m.pos_y,j))
        # We need to take a random sample from the list synapseList 
        numNewSynapses = self.newSynapseCount-len(c.cells[i].segments[s].synapses)
        # Make sure that the number of new synapses to choose isn't larger than the
        #total amount of active synapses to choose from but is larger than zero.
        if numNewSynapses>len(synapseList):
            numNewSynapses = len(synapseList)
        if numNewSynapses<0:
            numNewSynapses = 0
            #print "%s new synapses from len(synList) = %s" %(numNewSynapses,len(synapseList))
            # return an empty list. This means this segment has too many synapses already
            return np.array([])
        return np.array(random.sample(synapseList,numNewSynapses))
    def getActiveSegment(self,c,i,t,state):
        # Returns a sequence segment if there are none then returns the most active segment 
        highestActivity = 0
        for s in c.cells[i].segments:
            activeState = 1
            activity = self.segmentActive(s,t,activeState)
            if s.sequenceSegment == t:
                print "RETURNED SEQUENCE SEGMENT"
                return s
            else:
                mostActiveSegment = s
                if activity > highestActivity:
                    highestActivity = activity
                    mostActiveSegment =  s
        return mostActiveSegment
    def segmentActive(self,s,timeStep,state):
        # For Segment s check if the number of synapses with the state "state" is larger then 
        # the self.activationThreshold
        # state is -1 = predictive state, 1 = active, 2 = learn state 
        count = 0
        for i in range(len(s.synapses)):
            x = s.synapses[i].pos_x
            y = s.synapses[i].pos_y
            cell = s.synapses[i].cell
            if state == 1:  # 1 is active state
                if self.columns[y][x].activeStateArray[cell] == timeStep:
                    count += 1
            elif state == -1: # -1 is predictive state
                if self.columns[y][x].predictiveStateArray[cell] == timeStep:
                    count += 1
            elif state == 2:    # 2 is learn state
                if self.columns[y][x].learnStateArray[cell] == timeStep:
                    count += 1
            else:
                print "ERROR state is not a -1 predictive or 1 active or 2 learn"
        if count > self.activationThreshold: 
            #print"         %s synapses were active on segment"%count
            # If the state is active then those synapses in the segment have activated the
            #segment as being a sequence segment i.e. the segment is predicted to activate the cell
            #on the next time step.
            if state == 1:  # 1 is active state
                s.sequenceSegment=timeStep
            return True
        else:
            return False
    def segmentNumSynapsesActive(self,s,timeStep):
        # For Segment s find the number of active synapses. Synapses whose end is on
        # an active cell
        count = 0
        for i in range(len(s.synapses)):
            x = s.synapses[i].pos_x
            y = s.synapses[i].pos_y
            cell = s.synapses[i].cell
            if self.columns[y][x].activeStateArray[cell] == timeStep:
                count += 1
        return count
    def getBestMatchingSegment(self,c,i,t):
        # This routine is agressive. The permanence value is allowed to be less
        # then connectedPermance and activationThreshold > number of active Synpses > minThreshold
        h = 0 # mostActiveSegmentIndex
        # Look through the segments for the one with the most active synapses
        print "getBestMatchingSegment for x,y,c = %s,%s,%s num segs = %s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments))
        for g in range(len(c.cells[i].segments)):
            currentSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[g],t)
            mostActiveSegSynCount = self.segmentNumSynapsesActive(c.cells[i].segments[h],t)
            if  currentSegSynCount>mostActiveSegSynCount:
                h = g
                print "\n new best matching segment found for h = %s\n"%h
                #print "segIndex = %s num of syn = %s num active syn = "%(h,len(c.cells[i].segments[h].synapses),currentSegSynCount)
                print "segIndex = %s"%(h)
        if self.segmentNumSynapsesActive(c.cells[i].segments[h],t)>self.minThreshold:
            print "returned the segment index (%s) which HAD MORE THAN THE MINTHRESHOLD SYNAPSES"%h
            return h    # returns just the index to the most active segment in the cell
        print "returned no segment. None had enough active synapses return -1"
        return -1   # -1 means no segment was active enough and a new one will be created.
    def getBestMatchingCell(self,c,timeStep):
        # Return the cell and the segment that is most matching in the column.
        # If no cell has a matching segment (no segment has more then minThreshold synapses active)
        # then return the cell with the fewest segments
        bestCell = -1   # Cell index with the most active Segment
        bestSegment = -1 # The segment index for the most active segment
        fewestSegments = 0 # The cell index of the cell with the least munber of segments
        h = -1           # h is the SegmentIndex of the most active segment for the current cell i  
        print "getBestMatchingCell for x,y = %s,%s"%(c.pos_x,c.pos_y)
        for i in range(self.cellsPerColumn):
            # We minus one since the length of the segments array starts at 1 
            #and fewestSegments index starts at 0.
            if len(c.cells[i].segments) < len(c.cells[fewestSegments].segments):
                fewestSegments = i
            h = self.getBestMatchingSegment(c,i,timeStep)
            if h >= 0:
                if self.segmentNumSynapsesActive(c.cells[i].segments[h],timeStep) > self.segmentNumSynapsesActive(c.cells[bestCell].segments[bestSegment],timeStep):
                    bestCell = i
                    bestSegment = h
        if bestCell != -1:
            print "returned from GETBESTMATCHINGCELL the cell with the best segment = %s"%bestCell
            return (bestCell,bestSegment)
        else:
            # Return the first segment from the cell with the fewest segments
            print "returned from getBestMatchingCell the cell %s with the fewest segments"%fewestSegments
            return (fewestSegments,0)
    def getSegmentActiveSynapses(self,c,i,t,s,newSynapses=False):
        # Returns an segmentUpdate structure. This is used to update the segments and there
        # synapses during learning. It adds the synapses from the segments synapse list
        # that have an active end, to the segmentUpdate structure so these synapses can be updated
        # appropriately (either inc or dec) later during learning.
        # s is the index of the segment in the cells segment list.
        newSegmentUpdate = {'index':s,'activeSynapses':np.array([],dtype=object),'newSynapses':np.array([],dtype=object),'sequenceSegment':0, 'createdAtTime': t}
        print "    getSegmentActiveSynapse called for t = %s x,y,i = %s,%s,%s newSyn = %s"%(t,c.pos_x,c.pos_y,i,newSynapses) 
        # If segment exists then go through an see which synapses are active. 
        #Add them to the update structure.
        if s != -1:
            if len(c.cells[i].segments) > 0:      # Make sure the array isn't empty
                if len(c.cells[i].segments[s].synapses) > 0:
                    for k in range(len(c.cells[i].segments[s].synapses)):
                        end_x = c.cells[i].segments[s].synapses[k].pos_x
                        end_y = c.cells[i].segments[s].synapses[k].pos_y
                        end_cell = c.cells[i].segments[s].synapses[k].cell
                        #print "Synapse ends at (%s,%s,%s)" %(end_x,end_y,end_cell)
                        # Check to see if the Synapse is connected to an active cell
                        if self.columns[end_y][end_x].activeStateArray[end_cell] == t:
                            # If so add it to the updateSegment structure
                            print "     active synapse starts at x,y,cell,segIndex = %s,%s,%s,%s"%(c.pos_x,c.pos_y,i,s)
                            print "     the synapse ends at x,y,cell = %s,%s,%s"%(end_x,end_y,end_cell)
                            newSegmentUpdate['activeSynapses']=np.append(newSegmentUpdate['activeSynapses'],c.cells[i].segments[s].synapses[k])
        if newSynapses == True:
            # Add new synapses that have an active end. 
            # We add them  to the segmentUpdate structure. They are added
            # To the actual segment later during learning when the cell is 
            # in a learn state.
            newSegmentUpdate['newSynapses']=np.append(newSegmentUpdate['newSynapses'],self.randomActiveSynapses(c,i,s,t))
            print "     New synapse added to the segmentUpdate"
        # Return the update structure.
        print "     returned from getSegmentActiveSynapse"
        return newSegmentUpdate
    def adaptSegments(self,c,i,positiveReinforcement):
        print " adaptSegments x,y,cell = %s,%s,%s positive reinforcement = %r"%(c.pos_x,c.pos_y,i,positiveReinforcement)
        # Adds the new segments to the cell and inc or dec the segments synapses
        # If positive reinforcement is true then segments on the update list
        # get their permanence values increased all others get their permanence decreased. 
        # If positive reinforcement is false then decrement the permanence value for the active synapses.  
        
        for j in range(len(c.cells[i].segmentUpdateList)):
            segIndex=c.cells[i].segmentUpdateList[j]['index']
            print "     segIndex = %s"%segIndex
            # If the segment exists
            if segIndex>-1:
                for s in c.cells[i].segmentUpdateList[j]['activeSynapses']:
                    # For each synapse in the segments activeSynapse list increment or
                    # decrement their permanence values.
                    # The synapses in the update segment structure are already in the segment. The
                    # new synapses are not yet however.
                    if positiveReinforcement==True:
                        s.permanence += c.permanenceInc
                        s.permanence = min(1.0,s.permanence)
                    else:
                        s.permanence -= c.permanenceDec
                        s.permanence = max(0.0,s.permanence)
                    print "     x,y,cell,segment= %s,%s,%s,%s syn end x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,c.cells[i].segmentUpdateList[j]['index'],s.pos_x,s.pos_y,s.cell)
                    print "     synapse permanence = %s was increased"%(s.permanence)
                # Decrement the permanence of all synapses in the synapse list
                for s in c.cells[i].segments[segIndex].synapses:
                    s.permanence -= c.permanenceDec
                    s.permanence = max(0.0,s.permanence)
                    #print "     x,y,cell,segment= %s,%s,%s,%s syn end x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i,j,s.pos_x,s.pos_y,s.cell)
                    #print "     synapse permanence = %s"%(s.permanence)
                # Add the new Synpases in the structure to the real segment
                #print c.cells[i].segmentUpdateList[j]['newSynapses']
                #print "oldActiveSyn = %s newSyn = %s" %(len(c.cells[i].segments[segIndex].synapses),len(c.cells[i].segmentUpdateList[j]['newSynapses']))
                c.cells[i].segments[segIndex].synapses=np.append(c.cells[i].segments[segIndex].synapses,c.cells[i].segmentUpdateList[j]['newSynapses'])
                # Delete synapses that have low permanence values.
                self.deleteWeakSynapses(c,i,segIndex)
            # If the segment is new (the segIndex = -1) add it to the cell
            else:
                newSegment = Segment()
                newSegment.synapses = c.cells[i].segmentUpdateList[j]['newSynapses']
                c.cells[i].segments = np.append(c.cells[i].segments,newSegment)
                print "     new segment added for x,y,cell,seg = %s,%s,%s,%s"%(c.pos_x,c.pos_y,i,len(c.cells[i].segments)-1)
        c.cells[i].segmentUpdateList=[]  # Delete the list as the updates have been added.
        
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
        #print "actve cols before %s" %self.activeColumns
        for i in range(len(self.columns)):
            for c in self.columns[i]:
                if c.overlap>0:
                    minLocalActivity = self.kthScore(self.neighbours(c),self.desiredLocalActivity)  
                    #print "current column = (%s,%s)"%(c.pos_x,c.pos_y)
                    if c.overlap>=minLocalActivity:
                        self.activeColumns=np.append(self.activeColumns,c)
                        c.activeState = True
                        #print "x,y = %s,%s overlap = %d min = %d" %(c.pos_x,c.pos_y,c.overlap,minLocalActivity)
                else:
                    c.activeState = False
                self.updateActiveDutyCycle(c)       # Update the active duty cycle variable of every column
    def learning(self):
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
    def updateActiveState(self,timeStep):
        # First function called to update the temporal pooler.
        # First reset the active cells calculated from the previous time step. 
        for c in self.activeColumns:
            buPredicted = False
            lcChosen = False
            print "\n ACTIVE COLUMN x,y = %s,%s time = %s"%(c.pos_x,c.pos_y,timeStep)
            for i in range(self.cellsPerColumn):
                if self.predictiveState(c,i,timeStep-1) == True:
                    activeState = 1
                    s = self.getActiveSegment(c,i,timeStep-1,activeState)
                    # Since we get the active segments from 1 time step ago then we need to 
                    #find which of these where sequence segments 1 time step ago. This mean they
                    #were predicting that the cell would be active now.
                    if s.sequenceSegment == timeStep-1:
                        buPredicted = True
                        c.activeStateArray[i] = timeStep
                        learnState = 2
                        if self.segmentActive(s,timeStep-1,learnState) == True:
                            lcChosen = True
                            c.learnStateArray[i] = timeStep
            if buPredicted == False:
                print "No cell in this column predicted"
                for i in range(self.cellsPerColumn):
                    c.activeStateArray[i] = timeStep
            if lcChosen == False:
                print "lcChosen Getting the best matching cell to set as learning cell"
                (cell,s) = self.getBestMatchingCell(c,timeStep)
                c.learnStateArray[cell] = timeStep
                sUpdate = self.getSegmentActiveSynapses(c,cell,timeStep-1,s,True)
                sUpdate['sequenceSegment']=timeStep
                c.cells[cell].segmentUpdateList.append(sUpdate)
    def updatePredictiveState(self,timeStep):
        # The second function call for the temporal pooler. 
        # Updates the predictive state of cells.
        print "\n SECOND FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    segIndex=0   
                    for s in c.cells[i].segments:
                        activeState = 1
                        if self.segmentActive(s,timeStep,activeState) == True:
                            print "time = %s segment x,y,cell,segindex = %s,%s,%s,%s is active and NOW PREDICTING"%(timeStep,c.pos_x,c.pos_y,i,segIndex)
                            c.predictiveStateArray[i] = timeStep
                            activeUpdate=self.getSegmentActiveSynapses(c,i,timeStep,segIndex,False)
                            c.cells[i].segmentUpdateList.append(activeUpdate)
                            # This differs to the CLA. All our segments are only active
                            # when in a predicting state so we don't need predSegment.
                            #predSegment=self.getBestMatchingSegment(c,i,timeStep-1)
                            #predUpdate=self.getSegmentActiveSynapses(c,i,timeStep-1,predSegment,True)
                            #c.cells[i].segmentUpdateList.append(predUpdate)
                        segIndex=segIndex+1  # Need this to hand to getSegmentActiveSynapses
    def temporalLearning(self,timeStep):
        # Third function called for the temporal pooler.
        # The update structures are implemented on the cells
        print "\n    THIRD FUNCTION "
        for k in range(len(self.columns)):
            for c in self.columns[k]:
                for i in range(len(c.cells)):
                    #if c.predictiveStateArray[i] != 0.0:   # For debugging
                    #    print "predictiveStateArray for x,y,i = %s,%s,%s is time = %s"%(c.pos_x,c.pos_y,i,c.predictiveStateArray[i])
                    if c.learnStateArray[i] == timeStep:
                        #print "learn state for x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c,i,True)
                    #if self.predictiveState(c,i,timeStep) == False and self.predictiveState(c,i,timeStep-1) == True:
                        #print "predictive state for x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        #self.adaptSegments(c,i,False)
                    ## Trying a different method to the CLA white pages
                    if self.activeState(c,i,timeStep) == False and self.predictiveState(c,i,timeStep-1) == True:
                        #print "predictive state for x,y,cell = %s,%s,%s"%(c.pos_x,c.pos_y,i)
                        self.adaptSegments(c,i,False)
        
class HTM:
    def __init__(self, numLayers,input, column_array_width,column_array_height):
        self.quit = False
        # The class contains multiple HTM layers stacked on one another
        self.numberLayers = numLayers   # The number of layers in the HTM network
        self.width = column_array_width
        self.height = column_array_height
        self.HTMLayerArray = np.array([],dtype = object)
        for i in range(numLayers):
            self.HTMLayerArray = np.append(self.HTMLayerArray,HTMLayer(input,self.width,self.height))
    def spatial_temporal(self,input):
        # Update the spatial and temporal pooler. Find spatial and temporal patterns from the input.
        # This updates the columns and all there vertical synapses as well as the cells and the horizontal Synapses.
        for i in range(len(self.HTMLayerArray)):
            self.HTMLayerArray[i].timeStep = self.HTMLayerArray[i].timeStep+1
            if i == 0:
                self.HTMLayerArray[i].Input(input)
            else:
                output = self.HTMLayerArray[i-1].output
                self.HTMLayerArray[i].Input(output)
            # This updates the spatial pooler
            self.HTMLayerArray[i].Overlap()
            self.HTMLayerArray[i].inhibition()
            self.HTMLayerArray[i].learning()
            #This Updates the temporal pooler
            self.HTMLayerArray[i].updateActiveState(self.HTMLayerArray[i].timeStep)
            self.HTMLayerArray[i].updatePredictiveState(self.HTMLayerArray[i].timeStep)
            self.HTMLayerArray[i].temporalLearning(self.HTMLayerArray[i].timeStep)

# NOT CURRENTLY USED WITH THE VIEWER
def runLoop(HTM,input,iteration):
    #HTM_draw.initialize_drawing()
    print "NEW learning stage\n"
    # Created an alternating pattern to learn with noise for testing
    # Zero all inputs
    for k in range(len(input)):
        for l in range(len(input[k])):
            input[k][l] = 0
            # Add some noise
            some_number = round(random.uniform(0,10))
            if some_number>8:
                input[k][l] = 1
    if iteration % 2 == 0:
        print "pattern1"
        input[2][3:8] = 1
        input[3][7] = 1 
        input[4][7] = 1 
        input[5][7] = 1 
        input[6][3:8] = 1
        input[3][3] = 1 
        input[4][3] = 1 
        input[5][3] = 1
        input[6][3] = 1 
    else:
        if iteration<80 or iteration>150:
            print "pattern2"
            input[8][7:9] = 1
            input[7][7:9] = 1
        else:
            print "pattern3"
            input[9][4:9] = 1
            input[8][8] = 1 
            input[7][8] = 1 
            input[6][8] = 1 
            input[5][4:9] = 1
            input[6][4] = 1 
            input[7][4] = 1 
            input[8][4] = 1
            input[9][4] = 1 
    #Learning and updating
    #HTM.spatial_temporal()
    #HTM_draw.draw_HTM(HTM,input)   # For drawing using the old 2D HTM_draw.py
    #HTM_draw.quit_drawing()

#if __name__ == "__main__":
#    sizew = 12
#    sizeh = 10
#    numLayers = 3
#    input = np.array([[round(random.uniform(0,1)) for i in range(sizew)] for j in range(sizeh)])
#    HTMNetwork = HTM(numLayers,input,sizew,sizeh)
#    run_loop(HTMNetwork,input)