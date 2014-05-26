#!/usr/bin/python

"""
HTM poker GUI
author: Calum Meiklejohn
website: calumroy.com
last edited: June 2013

This code draws the input and a HTM network in a grid using PyQt

It creates a simplified version of an Inverted_Pendulumn and 
attempts to learn and control the system using the HTM network.

"""
import sys
sys.path.insert(0, './')       #Add the parent diectory to the path to search for modules
import numpy as np
import random
import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QObject, pyqtSlot
import HTM_Balancer as HTM_V
import math
import copy

import Inverted_Pendulum as invertPen

class levelPopup(QtGui.QWidget):
    
    # Create a signal to tell the network which level was selected
    levelSelectedSignal = QtCore.pyqtSignal(int)
    
    def __init__(self,numLevels):
        QtGui.QWidget.__init__(self)
        self.numLevels = numLevels
        layout = QtGui.QVBoxLayout()
        self.checks = []
        for i in range(self.numLevels):
            c = QtGui.QCheckBox("Level %s" %i)
            c.stateChanged.connect(self.levelSelected)
            layout.addWidget(c)
            self.checks.append(c)
        self.setLayout(layout)
    def levelSelected(self,i):
        # Check each check box to find out which one was selected
        for i in range(len(self.checks)):
            if self.checks[i].isChecked()==True:
                self.levelSelectedSignal.emit(i)
                #print"levelSelectedSignal sent"
    def paintEvent(self, event):
        dc = QtGui.QPainter(self)

class popup(QtGui.QWidget):
    
    # Create a signal to tell the scene which segment was selected
    segmentSelectedSignal = QtCore.pyqtSignal(int)
    
    def __init__(self,x,y,cell,numSegments):
        QtGui.QWidget.__init__(self)
        self.pos_x = x
        self.pos_y = y
        self.cell = cell
        self.numSegments = numSegments
        layout = QtGui.QVBoxLayout()
        self.checks = []
        for i in range(self.numSegments):
            c = QtGui.QCheckBox("segment %s" %i)
            c.stateChanged.connect(self.segmentSelected)
            layout.addWidget(c)
            self.checks.append(c)
        self.setLayout(layout)
        
    def segmentSelected(self,i):
        # Check each check box to find out which one was selected
        for i in range(len(self.checks)):
            if self.checks[i].isChecked()==True:
                self.segmentSelectedSignal.emit(i)
                #print"segmentSelectedSignal sent"
                #self.close()

    def paintEvent(self, event):
        dc = QtGui.QPainter(self)


class HTMColumn(QtGui.QGraphicsRectItem):
    def __init__(self,HTM_x,HTM_y,squareSize,pen,brush):
        super(HTMColumn, self).__init__()
        self.initUI(HTM_x,HTM_y,squareSize,pen,brush)

    def initUI(self,HTM_x,HTM_y,squareSize,pen,brush):
        self.pos_x = HTM_x  # The x position in the HTM grid
        self.pos_y = HTM_y # The y position in the HTM grid
        self.setPos(HTM_x*squareSize,HTM_y*squareSize)
        self.setRect(0,0,squareSize,squareSize)
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        
    #def mousePressEvent(self,event):
    #    print"column pos_x,pos_y = %s,%s"%(self.pos_x,self.pos_y)



class HTMCell(QtGui.QGraphicsRectItem):
    def __init__(self,HTM_x,HTM_y,cell,x,y,squareSize,pen,brush):
        super(HTMCell, self).__init__()
        self.initUI(HTM_x,HTM_y,cell,x,y,squareSize,pen,brush)

    def initUI(self,HTM_x,HTM_y,cell,x,y,squareSize,pen,brush):
        self.pos_x = HTM_x  # The x position in the HTM grid
        self.pos_y = HTM_y # The y position in the HTM grid
        self.cell = cell  # The cell number oin the HTM grid
        self.setPos(x,y)
        self.setRect(0,0,squareSize,squareSize)
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
    
    #def mousePressEvent(self,event):
    #    print"cell pos_x,pos_y,cell = %s,%s,%s"%(self.pos_x,self.pos_y,self.cell)
        


class HTMInput(QtGui.QGraphicsView):

    def __init__(self,width,height):
        super(HTMInput, self).__init__()
        
        self.initUI(width,height)

    def initUI(self,width,height):
        self.scene=QtGui.QGraphicsScene(self)
        self.scaleSize = 1
        self.setScene(self.scene)
        self.size = 20
        self.cols = width
        self.rows = height
        self.pos_x = 0
        self.pos_y = 0
        self.inputArray = np.array([[0 for i in range(width)] for j in range(height)])
        self.drawGrid(self.rows,self.cols,self.size)
        self.show()
        
    def scaleScene(self,scaleSize):
        self.scale(scaleSize, scaleSize)

    def drawGrid(self, rows, cols, size):
        pen   = QtGui.QPen(QtGui.QColor(QtCore.Qt.black))
        brush = QtGui.QBrush(pen.color().darker(150))
        for y in range(rows):
                for x in range(cols):
                        value = self.inputArray[y][x]
                        if value == 0:
                            brush.setColor(QtCore.Qt.red);
                        else:
                            brush.setColor(QtCore.Qt.green);
                        item = self.scene.addRect(x*size,y*size,size,size, pen, brush)
                        item.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
    
    def updateInput(self):
        for y in range(self.rows):
                for x in range(self.cols):
                        brush = QtGui.QBrush(QtCore.Qt.green)
                        brush.setStyle(QtCore.Qt.SolidPattern)
                        item = self.scene.itemAt(x*self.size,y*self.size)
                        value = self.inputArray[y][x]
                        if value == 0:
                            brush.setColor(QtCore.Qt.red)
                            item.setBrush(brush)
                        else:
                            brush.setColor(QtCore.Qt.green)
                            item.setBrush(brush)
        
    def setInput(self,newInput):
        self.cols = len(newInput[0])
        self.rows = len(newInput)
        #print "rows, cols = (%s,%s)" %(self.rows, self.cols)
        self.inputArray = newInput
        
class HTMGridViewer(QtGui.QGraphicsView):
    
    def __init__(self,width,height,commandRow,numLevels):
        super(HTMGridViewer, self).__init__()
        self.initUI(width,height,commandRow,numLevels)

        
    def initUI(self,width,height,commandRow,numLevels):
        self.scene=QtGui.QGraphicsScene(self)
        self.scaleSize = 1
        self.setScene(self.scene)
        self.size = 20
        self.cols = width
        self.rows = height
        self.pos_x = 0
        self.pos_y = 0
        self.numCells = 3
        self.numLevels = numLevels
        self.level = 0  # Draw this level (Region) of the HTMNetwork
        self.layer = 0 # Draw this HTN layer in the level.
        self.commandRow = commandRow # The row where the command space starts.
        self.numCommRows = self.rows-self.commandRow 
        # The minimum number of cells that are active and where predicted for the command to be considered successful
        self.minNumberPredCells = 3 
        # For the popup segment selection box
        self.segmentSelect = None
        self.selectedItem = None
        # Keep track of the right mouse button to move the view
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self._mousePressed = False
        self._dragPos = None
        
        # Create HTM network with an empty input
        input = np.array([[0 for i in range(width)] for j in range(height)])
        self.htm = HTM_V.HTM(self.numLevels,input,width,height,self.numCells,commandRow)
        self.showAllHTM = True  # A flag to indicate to draw all the cells and column states
        
        self.showActiveCells = True
        self.showPredictCells = False
        self.showLearnCells = False
        
        self.scaleGridSize()
        self.cellItems = []   # Stores all the cell items in the scene
        self.columnItems = [] # Stores all the column items in the scene

        self.drawGrid(self.rows,self.cols,self.size)
        self.show()

    def selectedSegmentIndex(self,index):
        #print"Selected item pos_x,pos_y,cell,segment%s,%s,%s,%s"%(self.selectedItem.pos_x,self.selectedItem.pos_y,self.selectedItem.cell,index)
        self.drawSingleCell(self.selectedItem.pos_x,self.selectedItem.pos_y,self.selectedItem.cell,index)
        
    def scaleGridSize(self):
        # Scale the size of the grid so the cells can be shown if there are too many cells
        while (int (math.ceil(self.numCells ** 0.5))>self.size/2):
            self.size = self.size *2
            
    def drawGrid(self,rows, cols, size):
        # Used to initialise the graphics scene with columns and cells
        pen   = QtGui.QPen(QtGui.QColor(QtCore.Qt.black))
        brush = QtGui.QBrush(pen.color().darker(150))
        for x in range(cols):
                for y in range(rows):
                    value = self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[y][x].activeState
                    if value == False:
                            brush.setColor(QtCore.Qt.red)
                    if value == True:
                            brush.setColor(QtCore.Qt.green)
                    # Create a column item and add it to a list so we can iterate through them to update
                    columnItem = HTMColumn(x,y,size,pen,brush)
                    columnItem.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)  
                    #columnItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)                  
                    self.columnItems.append(columnItem) 
                    self.scene.addItem(columnItem)
                    self.drawCells(self.numCells,x,y,size)

    def drawCells(self,numCells,pos_x,pos_y,size):
        # Used to initialise the graphics scene with cells
        transp = QtGui.QColor(0, 0, 0, 0)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        pen   = QtGui.QPen(transp)
        brush = QtGui.QBrush(red)  # Color has an opacity
        # Find the smallest number which when squared is larger than numCells
        numSquares = int (math.ceil(numCells ** 0.5))
        # Set the small sqaures to a size smaller than the large ones
        squareSize = size/(1.5*numSquares)
        # Count the cells that are drawn so we can identify them
        count = 0
        for i in range(numSquares):
            for j in range(numSquares):
                brush = QtGui.QBrush(transpRed) # Make the non existent cells faint
                if count < numCells:
                    # Separate the small squares
                    x = pos_x*size + 0.5*squareSize+1.5*i*squareSize
                    y = pos_y*size + 0.5*squareSize+1.5*j*squareSize
                    cellItem = HTMCell(pos_x,pos_y,count,x,y,squareSize,pen,brush)
                    #cellItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
                    self.cellItems.append(cellItem) #Add the cells to a list so we can iterate through them to update
                    self.scene.addItem(cellItem)    # Add the cells to the scene
                # Increase the count to keep track of how many cells have been created
                count += 1
    def drawSingleColumn(self,pos_x,pos_y):
        # Draw the cells connected to the selected segment
        print "Column Synapse permanence"
        transp = QtGui.QColor(0, 0, 0, 0)
        #pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        # Go through each column. If it is in the synapse list draw it otherwise don't
        for i in range(len(self.columnItems)):
            column_pos_x=self.columnItems[i].pos_x
            column_pos_y=self.columnItems[i].pos_y
            brush = QtGui.QBrush(transpBlue)   # Have to create a brush with a color
            # Check each synapse and draw the connected columns
            for syn in self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x].connectedSynapses:
                if syn.pos_x==column_pos_x and syn.pos_y==column_pos_y:
                    print "     syn x,y= %s,%s Permanence = %s"%(column_pos_x,column_pos_y,syn.permanence)
                    brush.setColor(darkGreen);
            self.columnItems[i].setBrush(brush)
            #self.columnItems[i].setPen(pen)
                
    def drawSingleCell(self,pos_x,pos_y,cell,segment):
        # Draw the cells connected to the selected segment
        print"pos_x,pos_y,cell,seg = %s,%s,%s,%s"%(pos_x,pos_y,cell,segment)
        print "Segment Synapse permanence"
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        # Go through each cell. If it is in the synapse list draw it otherwise don't
        for i in range(len(self.cellItems)):
            cell_pos_x=self.cellItems[i].pos_x
            cell_pos_y=self.cellItems[i].pos_y
            cell_cell=self.cellItems[i].cell
            column = self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x]
            brush = QtGui.QBrush(transpBlue)   # Have to create a brush with a color
            # Check each synapse and draw the connected cells
            for syn in column.cells[cell].segments[segment].synapses:
                if syn.pos_x==cell_pos_x and syn.pos_y==cell_pos_y and syn.cell==cell_cell:
                    print "     syn x,y,cell= %s,%s,%s Permanence = %s, active times = %s"%(cell_pos_x,cell_pos_y,cell_cell,syn.permanence,column.activeStateArray[syn.cell])
                    brush.setColor(blue);
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)
                
    
    def updateHTMGrid(self):
        for i in range(len(self.columnItems)):
            brush = QtGui.QBrush(QtCore.Qt.green)
            brush.setStyle(QtCore.Qt.SolidPattern)
            pos_x=self.columnItems[i].pos_x
            pos_y=self.columnItems[i].pos_y
            value = self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x].activeState
            if value == 0:
                    brush.setColor(QtCore.Qt.red)
                    self.columnItems[i].setBrush(brush)
            if value == 1:
                    brush.setColor(QtCore.Qt.green)
                    self.columnItems[i].setBrush(brush)
        self.updateCells()
            
                    
    def updateCells(self):
        # Redraw the cells.
        timeStep=self.htm.HTMRegionArray[self.level].layerArray[self.layer].timeStep
        print " current levels TimeStep=%s"%(timeStep)
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        black = QtGui.QColor(0, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        for i in range(len(self.cellItems)):
            pos_x=self.cellItems[i].pos_x
            pos_y=self.cellItems[i].pos_y
            cell=self.cellItems[i].cell
            brush = QtGui.QBrush(transp) # Make the non existent cells faint
            if self.showActiveCells==True:
                if int(self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x].activeStateArray[cell,0]) == timeStep:
                    brush.setColor(blue);
                else:
                    brush.setColor(transpBlue);
            if self.showPredictCells==True:
                if int(self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x].predictiveStateArray[cell,0]) == timeStep:
                    brush.setColor(black);
                else:
                    brush.setColor(transpBlue);
            if self.showLearnCells==True:
                if int(self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x].learnStateArray[cell,0]) == timeStep:
                    brush.setColor(darkGreen);
                else:
                    brush.setColor(transpBlue);
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)
                
    
    def scaleScene(self,scaleSize):
        self.scale(scaleSize, scaleSize)
        
    def mousePressEvent(self,event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self._mousePressed = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._dragPos = event.pos()
            event.accept()
            item=self.itemAt(event.x(),event.y())
            if item.__class__.__name__ == "HTMCell":
                print "cell"
                print "pos_x,pos_y,cell = %s,%s,%s"%(item.pos_x,item.pos_y,item.cell)
                numSegments = len(self.htm.HTMRegionArray[self.level].layerArray[self.layer].columns[item.pos_y][item.pos_x].cells[item.cell].segments)
                self.selectedItem = item
                item_pos=item.pos()
                popup_pos_x=item_pos.x()+self.x()
                popup_pos_y=item_pos.y()+self.y()
                # Create the popup window at a certain position
                self.segmentSelect = popup(event.x(),event.y(),item.cell,numSegments)
                self.segmentSelect.setGeometry(QtCore.QRect(popup_pos_x, popup_pos_y, 200, 200))
                # Create and connect a Slot to the signal from the check box
                self.segmentSelect.segmentSelectedSignal.connect(self.selectedSegmentIndex)
                self.segmentSelect.show()
            if item.__class__.__name__ == "HTMColumn":
                print"column"
                print "pos_x,pos_y = %s,%s"%(item.pos_x,item.pos_y)
                # Draw the columns synapses.
                self.drawSingleColumn(item.pos_x,item.pos_y)      
        if event.buttons() == QtCore.Qt.RightButton:
            # Toggle the view from predicted, learn and active cells.
            if self.showActiveCells==True:
                self.showActiveCells = False
                self.showPredictCells = True
                self.showLearnCells = False
            elif self.showPredictCells==True:
                self.showActiveCells = False
                self.showPredictCells = False
                self.showLearnCells = True
            elif self.showLearnCells==True:
                self.showActiveCells = False
                self.showPredictCells = False
                self.showLearnCells = False
            else:
                self.showActiveCells = True
            self.updateHTMGrid()

               
    def mouseMoveEvent(self, event):
        if self._mousePressed:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        else:
            super(HTMGridViewer, self).mouseMoveEvent(event)        

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self.setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
            #self._mousePressed = False
        #if event.button() == QtCore.Qt.RightButton:
        #        super(HTMGridViewer, self).mouseReleaseEvent(event)
        super(HTMGridViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event): pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control and not self._mousePressed:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        else:
            super(HTMGridViewer, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            if not self._mousePressed:
                self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            super(HTMGridViewer, self).keyPressEvent(event)


    def wheelEvent(self,  event):
        factor = 1.2;
        if event.delta() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)

    def step(self,input,level):
        self.htm.HTMRegionArray[level].spatialTemporal(input,self.layer)   
      
    def predictedCommand(self,level):
        # Return the predicted command as an array input.
        numberCols = len(self.htm.HTMRegionArray[level].layerArray[self.layer].columns[0])
        numberRows = len(self.htm.HTMRegionArray[level].layerArray[self.layer].columns)
        commandGrid = np.array([[0 for c in range(numberCols)] for r in range(numberRows)])
        htmLevel = self.htm.HTMRegionArray[level]
        currentTime=htmLevel.layerArray[self.layer].timeStep
        # only look at the cells in the command space.
        for k in range(self.htm.commandRow,len(self.htm.HTMRegionArray[level].layerArray[self.layer].columns)):
            for m in range(len(self.htm.HTMRegionArray[level].layerArray[self.layer].columns[k])):
                c = htmLevel.layerArray[self.layer].columns[k][m]
                for i in range(len(c.cells)):
                    # Set the output to true if a cell is in predictive stat for a particular column for the current timeStep
                    if htmLevel.layerArray[self.layer].predictiveState(c,i,currentTime)==True:
                        commandGrid[k][m] = 1
        return commandGrid    
    #def predictedCommand(self,level):
        # Return the predicted command.
        # The command which has the most predicted cells.
        # Commands are acceleration levels. 
        # left is negative acc right is positive acc
        # ONLY WORKS IF THE COMMANDS ARE ONE COLUMN WIDE. THIS MUST BE FIXED!
        # numberCols = len(self.htm.HTMLayerArray[level].columns[0])
        # accScore = np.array([0 for i in range(numberCols)])
        # htmLevel = self.htm.HTMLayerArray[level]
        # currentTime=htmLevel.timeStep
        # # only look at the cells in the command space.
        # for k in range(self.htm.commandRow,len(self.htm.HTMLayerArray[level].columns)):
        #     for m in range(len(self.htm.HTMLayerArray[level].columns[k])):
        #         c = htmLevel.columns[k][m]
        #         for i in range(len(c.cells)):
        #             # If the cell is in predictive state for the current timeStep
        #             if htmLevel.predictiveState(c,i,currentTime)==True:
        #                     accScore[m] += 1
        # # LikelyCommand is the acceleration command with the largest number of predicting cells.
        # highestScore = accScore[0]
        # sameCommands = []
        # likelyCommand = None
        # for m in range(len(accScore)):
        #     if accScore[m] > 0:
        #         if accScore[m] > highestScore:
        #             likelyCommand = m
        #             sameCommands = []
        #         if accScore[m] == likelyCommand:
        #             sameCommands.append(m)
        # # If two commands had the same number of predicting cells than choose a random command.
        # if len(sameCommands)>0:
        #     randomCommand = random.choice(sameCommands)
        #     # Convert the column number into an acceleration. Left is neg right is pos
        #     return -round(numberCols/2)+randomCommand
        # elif likelyCommand != None:
        #     return -round(numberCols/2)+likelyCommand
        # else:
        #     return 'none'


    def commandSuccessful(self,level):
        # Return whether the new input contains mostly active cells from non bursting columns.
        # If the number of non bursting cells is larger than the predefined threshold
        # then the previous command is said to have successfully produced a predicted result.
        numberRows = len(self.htm.HTMRegionArray[level].layerArray[self.layer].columns)
        notBurstingScore = 0
        endRow = (numberRows - self.htm.commandRow) # Only search through the new input space not the feedback or command space.
        htmLevel = self.htm.HTMRegionArray[level]
        currentTime=htmLevel.layerArray[self.layer].timeStep
        # only look at the cells in the input space.
        for col in htmLevel.layerArray[self.layer].activeColumns:
            if col.pos_y < endRow:
                numCellsActive = 0# Reset the count to work out if the column of cells is bursting
                for i in range(len(col.cells)):
                    # If the cell is in active state for the current timeStep
                    if htmLevel.layerArray[self.layer].activeState(col,i,currentTime)==True:
                            numCellsActive += 1 
                if numCellsActive == 1:  # The column is not bursting
                    notBurstingScore += 1
        if notBurstingScore > self.minNumberPredCells:
            return True
        else:
            return False
        
    def higherCommand(self,level):
        # Return the predicted command from the higher level.
        htmLevel = self.htm.HTMRegionArray[level]
        # The higher levels command space is the left half of the command space.
        fbCommandCols=self.cols/2
        # Create an empty array to store the feedback command
        fbCommand=np.array([[0 for i in range(fbCommandCols)] for j in range(self.numCommRows)])
        # If this is the highest level then return an empty array.
        # There is no feedback for the highest level.
        if level<self.htm.numLevels:
            for k in range(self.htm.commandRow,len(htmLevel.layerArray[self.layer].columns)):
                for m in range(0,fbCommandCols):
                    c = htmLevel.layerArray[self.layer].columns[k][m]
                    for i in range(len(c.cells)):
                        # Check each cell if it's in the predicted state for the \
                        # current timeStep of the level
                        if htmLevel.layerArray[self.layer].predictiveState(c,i,htmLevel.timeStep)==True:
                            row=k-self.htm.commandRow
                            col=m
                            fbCommand[row][col]=1
        return fbCommand

    def inSpaceOutput(self,level):
        # Return the active columns of the input space.
        return self.htm.HTMRegionArray[level].layerArray[self.layer].inSpaceOutput
        
        
class HTMNetwork(QtGui.QWidget):
    # Creates a HTM multi level network. Each level consists of an input space 
    # which is the lowest index rows and a command space which is the highest rows. 
    # These spaces are in the HTM level so synapses can form between their cells.
    # The command space is split in two. The left side has the current levels commands
    # and the right side is the feedback from the upper levels command space.
    
    def __init__(self):
        super(HTMNetwork, self).__init__()
        self.initUI()

    def initUI(self):
        self.iteration = 0
        self.origIteration = 0  # Stores the iteration for the previous saved HTM
        self.numLevels = 2 # The number of levels.
        self.angleInputHeight = 4   # How many rows will make up the angle input space.
        self.width = 9  # The number of columns making up the input spaces
        self.numCommRows = 4   # The number of rows that are command rows
        
        # Create the physics simualtion class
        self.invPen = invertPen.InvertedPendulum()
        self.angle = 3     # The angle that the inverted pendulum is at.
        self.angleOverlap = 0 # The number of columns that an angle position can overlap either side.
        self.minAngle = 1 # The angle that a cell in the first column represents.
        self.maxAngle = 9 # The angle that a cell in the last column represents.
        self.desAngle = 3 # The desired angle that the system should aim to acheive.
        self.oldAngle = np.array([self.angle for i in range(self.numLevels)])   # An array to store the previous angle value for each level
        #self.acceleration = 0.0   # The acceleration commanded by the HTM
        self.maxAcc = 1  # The maximum acceleration.
        self.minAcc = -1  # The maximum acceleration.


        self.command = np.array([0 for i in range(self.numLevels)])
        self.previousCommand = np.array([0 for i in range(self.numLevels)])
        self.height=self.angleInputHeight+2*self.numCommRows 
        
        # Set which row specifies the start of the command space.
        # This will be the row below the feedback commands
        # The feedback command space has the same size as the command space.
        self.commandRow = self.angleInputHeight+self.numCommRows  # This is where the command cells start.
        
        # The input space includes the feedback command space
        self.inputSpace = self.setInput(self.width,(self.commandRow))
        self.commandSpace = self.setInput(self.width,(self.height-self.commandRow))
        self.HTMNetworkGrid = HTMGridViewer(self.width,self.height,self.commandRow,self.numLevels)
        self.inputGrid = HTMInput(self.width,self.height)

        self.angleInput = invertPen.createInput(self.angle, self.width, self.angleInputHeight, self.angleOverlap, self.minAngle, self.maxAngle)

        # Store the commands from each level.

        # Used to create and save new views
        self.markedHTMViews = []
        
        self.scaleFactor=0.2    # How much to scale the grids by
        self.grid = None    # This is the layout holding the frames.
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        
        self.btn1=None # HTM +
        self.btn2=None # HTM -
        self.btn3=None # All HTM
        self.btn4=None # Nsteps
        self.btn5=None # Step
        self.btn6=None # In +
        self.btn7=None # In -
        self.btn8=None # Active
        self.btn9=None # Predict
        self.btn10=None # Learn
        self.btn11=None # Save
        self.btn12=None # Load
        self.btn13=None # Level select
        self.btn14=None # Mark state
        self.makeFrames()
        self.makeButtons()


        #self.setWindowTitle('Main window')
        self.show()

    def setInput(self,width,height):
        input = np.array([[0 for i in range(width)] for j in range(height)])
        return input
    

    def makeButtons(self):
        self.btn1 = QtGui.QPushButton("HTM +", self)
        self.btn1.clicked.connect(self.HTMzoomIn)
        self.btn2 = QtGui.QPushButton("HTM -", self)
        self.btn2.clicked.connect(self.HTMzoomOut)
        self.btn3 = QtGui.QPushButton("All HTM", self)
        self.btn3.clicked.connect(self.showAllHTM)
        self.btn4 = QtGui.QPushButton("n steps", self)
        self.btn4.clicked.connect(self.nSteps)
        self.btn5 = QtGui.QPushButton("step", self)
        self.btn5.clicked.connect(self.oneStep)
        self.btn6 = QtGui.QPushButton("In +", self)
        self.btn6.clicked.connect(self.inputZoomIn)
        self.btn7 = QtGui.QPushButton("IN -", self)
        self.btn7.clicked.connect(self.inputZoomOut)
        self.btn8 = QtGui.QPushButton("Active Cells", self)
        self.btn8.clicked.connect(self.showActiveCells)
        self.btn9 = QtGui.QPushButton("Predict Cells", self)
        self.btn9.clicked.connect(self.showPredictCells)
        self.btn10 = QtGui.QPushButton("Learn Cells", self)
        self.btn10.clicked.connect(self.showLearnCells)
        self.btn11 = QtGui.QPushButton("save", self)
        self.btn11.clicked.connect(self.saveHTM)
        self.btn12 = QtGui.QPushButton("load", self)
        self.btn12.clicked.connect(self.loadHTM)
        self.btn14 = QtGui.QPushButton("mark", self)
        self.btn14.clicked.connect(self.markHTM)
        
        # Create the level dropDown
        self.levelDropDown()
        
        # Add the dropdown menu to the screens top frame
        # addWidget(QWidget, row, column, rowSpan, columnSpan)
        self.grid.addWidget(self.btn5, 1, 2, 1, 1)
        self.grid.addWidget(self.btn4, 1, 5, 1, 1)
        self.grid.addWidget(self.btn11, 2, 1, 1, 1)
        self.grid.addWidget(self.btn12, 2, 2, 1, 1)
        self.grid.addWidget(self.btn8, 1, 6, 1, 1)
        self.grid.addWidget(self.btn9, 1, 7, 1, 1)
        self.grid.addWidget(self.btn10, 1, 8, 1, 1)
        self.grid.addWidget(self.btn3, 2, 5, 1, 1)
        
        self.grid.addWidget(self.btn6, 3, 1, 1, 1)
        self.grid.addWidget(self.btn7, 3, 2, 1, 1)
        self.grid.addWidget(self.btn1, 3, 5, 1, 1)
        self.grid.addWidget(self.btn2, 3, 6, 1, 1)
        self.grid.addWidget(self.btn14, 2, 6, 1, 1)
        
    def levelDropDown(self):
        # Create a drop down button to select the level in the HTM to draw.
        self.btn13 = QtGui.QToolButton(self)
        self.btn13.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        self.btn13.setMenu(QtGui.QMenu(self.btn13))
        self.setLevelAction = QtGui.QWidgetAction(self.btn13)
        self.levelList = levelPopup(self.HTMNetworkGrid.htm.numLevels)
        self.setLevelAction.setDefaultWidget(self.levelList)
        self.btn13.menu().addAction(self.setLevelAction)
        # Create and connect a Slot to the signal from the check box
        self.levelList.levelSelectedSignal.connect(self.setLevel)
        # Add the dropdown menu to the screens top frame
        self.grid.addWidget(self.btn13, 1, 1, 1, 1)
    def showAllHTM(self):
        # Draw the entire HTM netwrok. This is used if previously just a 
        # single cells segment connection was being shown
        self.HTMNetworkGrid.showAllHTM = True
        self.HTMNetworkGrid.updateHTMGrid()
    
    def markHTM(self):
        # Mark the current state of the HTM by creatng an new view to view the current state.
        self.markedHTMViews.append(HTMGridViewer(self.width,self.height,self.commandRow,self.numLevels))
        # Use the HTMGridVeiw object that has been appended to the end of the list
        self.markedHTMViews[-1].htm = copy.deepcopy(self.HTMNetworkGrid.htm)
        # Update the view settings
        self.markedHTMViews[-1].showActiveCells = self.HTMNetworkGrid.showActiveCells
        self.markedHTMViews[-1].showLearnCells = self.HTMNetworkGrid.showLearnCells
        self.markedHTMViews[-1].showPredictCells = self.HTMNetworkGrid.showPredictCells
        # Redraw the new view 
        self.markedHTMViews[-1].updateHTMGrid()
        

    def showActiveCells(self):
        # Toggle between showing the active cells or not
        if self.HTMNetworkGrid.showActiveCells==True:
            self.HTMNetworkGrid.showActiveCells=False
        else:
            self.HTMNetworkGrid.showActiveCells=True
            # Don't show the learning or predictive cells
            self.HTMNetworkGrid.showPredictCells=False
            self.HTMNetworkGrid.showLearnCells=False
            
        self.HTMNetworkGrid.updateHTMGrid()
    def showPredictCells(self):
        # Toggle between showing the predicting cells or not
        if self.HTMNetworkGrid.showPredictCells==True:
            self.HTMNetworkGrid.showPredictCells=False
        else:
            self.HTMNetworkGrid.showPredictCells=True
            # Don't show the learning or active cells
            self.HTMNetworkGrid.showActiveCells=False
            self.HTMNetworkGrid.showLearnCells=False
        self.HTMNetworkGrid.updateHTMGrid()
    def showLearnCells(self):
        # Toggle between showing the learning cells or not
        if self.HTMNetworkGrid.showLearnCells==True:
            self.HTMNetworkGrid.showLearnCells=False
        else:
            self.HTMNetworkGrid.showLearnCells=True
            # Don't show the learning or active cells
            self.HTMNetworkGrid.showActiveCells=False
            self.HTMNetworkGrid.showPredictCells=False
        self.HTMNetworkGrid.updateHTMGrid()
    def HTMzoomIn(self):
        #self.HTMNetworkGrid.scale = self.HTMNetworkGrid.scale*1.2
        #self.HTMNetworkGrid.update()
        self.HTMNetworkGrid.scaleScene(1+self.scaleFactor)
    def HTMzoomOut(self):
        #self.HTMNetworkGrid.scale = self.HTMNetworkGrid.scale*0.8
        #self.HTMNetworkGrid.update()
        self.HTMNetworkGrid.scaleScene(1-self.scaleFactor)
    def inputZoomIn(self):
        self.inputGrid.scaleScene(1+self.scaleFactor)
    def inputZoomOut(self):
        self.inputGrid.scaleScene(1-self.scaleFactor)

    def makeFrames(self):
        self.frame1 = QtGui.QFrame(self)
        self.frame1.setLineWidth(3)
        self.frame1.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        self.frame2 = QtGui.QFrame(self)
        self.frame2.setLineWidth(3)
        self.frame2.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        #self.frame3 = QtGui.QFrame(self)
        #self.frame3.setLineWidth(3)
        #self.frame3.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(1)
        # addWidget(QWidget, row, column, rowSpan, columnSpan)
        #self.grid.addWidget(self.HTMNetworkGrid,3,5,10,4)
        self.grid.addWidget(self.inputGrid,1,1,10,8)
        self.grid.addWidget(self.HTMNetworkGrid,8,1,10,8)
        #self.grid.addWidget(self.frame1, 1, 1, 2, 8)
        #self.grid.addWidget(self.frame2, 3, 1, 10, 4)
        #self.grid.addWidget(self.frame3, 3, 5, 10, 4)
        self.setLayout(self.grid)
    def setLevel(self,level):
        # Set the level for the HTMVeiwer to draw to the selected level.
        print "Level set to %s"%level
        self.HTMNetworkGrid.level = level
        # Update the columns and cells of the HTM viewer
        self.HTMNetworkGrid.updateHTMGrid()
        
    def saveHTM(self):
        self.HTMNetworkGrid.htm.saveRegions()
        self.origIteration = self.iteration
        print "Saved HTM layers"
    def loadHTM(self):
        # We need to make sure the GUI points to the correct object
        origHTM = self.HTMNetworkGrid.htm.loadRegions()
        self.HTMNetworkGrid.htm.HTMRegionArray = origHTM
        self.iteration = self.origIteration
        self.HTMNetworkGrid.iteration = self.origIteration
        print "loaded HTM layers"
     
    def oneStep(self):
        # Used as a call back for the one step button.
        # This is done so the True argument is passed and the viewers are then updated.
        self.step(True)
       
    def nSteps(self):
        numSteps, ok = QtGui.QInputDialog.getInt(self, 'number of steps','steps:')
        if ok:
            print numSteps
            # Minus one from the number of steps since we only update 
            # the veiwer on the last step.
            for i in range(numSteps-1):
                self.step(False)
            # Update the viewer on the last step.
            self.step(True)
                
            
    def step(self,updateViewer):
        # Update the inputs and run them through thte HTM levels just once.

        print "NEW PLAY. Current TimeStep = %s"%self.iteration
        # PART 1 MAKE NEW INPUT FOR LEVEL 0
        ############################################
        print "PART 1"

        # Return an average acceleration output to pass to the simulated inverted pendulum
        command = invertPen.medianAcc(self.HTMNetworkGrid.predictedCommand(0), self.minAcc, self.maxAcc)
        #print " CommandSpace = %s"%self.HTMNetworkGrid.predictedCommand(0)

        print " Predicted command is %s"%command
        if command=='none':
            command = random.randint(self.minAcc,self.maxAcc)
        print " command played was ",command      
        
        # Save level 0 command
        self.command[0]=command
        # update the inverted pendulum with the new acceleration command.
        self.oldAngle[0] = self.angle
        self.angle = self.invPen.step(command,self.minAngle, self.maxAngle, self.maxAcc ,1)  # Use 1 second for the step size.
        #print "self.angle=%s, self.width=%s, self.angleInputHeight=%s, self.angleOverlap=%s, self.minAngle=%s, self.maxAngle=%s"%(self.angle, self.width, self.angleInputHeight, self.angleOverlap, self.minAngle, self.maxAngle)
        self.angleInput = invertPen.createInput(self.angle, self.width, self.angleInputHeight, self.angleOverlap, self.minAngle, self.maxAngle)

        print " New angle = %s Old angle = %s"%(self.angle,self.oldAngle[0])
        
        
        # PART 2 ADD THE INPUT PARTS TOGETHER AND RUN THROUGH THE HTM LEVELS
        #####################################################################
        print "PART 2"
        self.iteration += 1 # Increase the time
        for lev in range(0,self.numLevels):
            # If the iteration is a power of 2 update the higher levels as well
            twoPowLev = math.pow(2,0)
            if self.iteration%twoPowLev==0:
                print " Level %s"%lev

                if lev>0:
                    # Get the output (of the input space) from the lower level (0)
                    # Don't include the feedback command space.
                    lowerLevelOutput = self.HTMNetworkGrid.inSpaceOutput(lev-1)[:-self.numCommRows]
                    #Check the lower levels input whether it was successful
                    lowerInSuccessful = self.HTMNetworkGrid.commandSuccessful(lev-1)
                else:
                    # The lowest level uses the newly created input 
                    lowerLevelOutput = self.inputSpace[:-self.numCommRows]
                    lowerInSuccessful = False

                
                # Check the last input and work out if the previous command was successful or not.
                print "     New angle = %s Old angle = %s"%(self.angle,self.oldAngle[lev])
                # This fuction is not called for the highest level.
                if lev==0:
                    commSuccessful = self.HTMNetworkGrid.commandSuccessful(lev)
                    if commSuccessful == True:
                        print "     level %s WON"%lev
                        newCommInput=invertPen.createInput(self.previousCommand[lev], self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                    else:
                        print "     level %s LOST"%lev
                        newCommInput=invertPen.createInput('none', self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                elif lowerInSuccessful==True:
                    # Then update the higher layer of the region as its command was successful
                    newCommInput=invertPen.createInput(self.command[lev], self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                    print "     level %s WON"%lev
                else:
                    newCommInput=invertPen.createInput('none', self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                    print "     level %s LOST"%lev


                ## The highest level checks if the pendulum has moved closer to the middle.
                ##if lev==(self.numLevels-1):
                #    # If the distance to the desired angle is smaller than before or is equal to zero then reinforce the command.
                #    if (abs(self.desAngle-self.angle) < abs(self.desAngle-self.oldAngle[lev]) or abs(self.desAngle-self.angle)==0):
                #        newCommInput=invertPen.createInput(self.command[lev], self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                #        print "     level %s WON"%lev
                #    else:
                #        newCommInput=invertPen.createInput('none', self.width, self.angleInputHeight, self.angleOverlap, self.minAcc, self.maxAcc)
                #        print "     level %s LOST"%lev


                # Get the command of the current level except for level 0
                # Level zero has already been done.
                # This must be done before the HTM is updated so the predicted
                # cells aren't over written.
                if lev!=0:
                    pred_command=invertPen.medianAcc(self.HTMNetworkGrid.predictedCommand(lev), self.minAcc, self.maxAcc)
                    print "     current levels pred command=%s"%(pred_command)
                    # If no command is predicted then use the last command.
                    # This command is sent down a level and used to learn against. 
                    if pred_command=='none':
                        self.command[lev] = random.randint(self.minAcc,self.maxAcc)
                    else:    
                        self.command[lev] = pred_command

                # Get the feed back from the upper level.
                # This fuction is not called for the highest level.
                if lev<(self.numLevels-1):
                    higherLevel=lev+1
                    #fbComm = self.HTMNetworkGrid.predictedCommand(higherLevel)
                    fbComm = self.command[higherLevel] 
                    print "     fbComm is %s"%fbComm
                    #print " fbComm=%s self.width=%s self.numCommRows=%s self.angleOverlap=%s self.minAcc=%s self.maxAcc=%s "%(fbComm, self.width, self.numCommRows, self.angleOverlap, self.minAcc, self.maxAcc)
                    upperCommInput = invertPen.createInput(fbComm, self.width, self.numCommRows, self.angleOverlap, self.minAcc, self.maxAcc)
                else:
                    # upperCommInput = np.array([[0 for i in range(self.width)] for j in range(self.numCommRows)])
                    upperCommInput = invertPen.createInput(0, self.width, self.numCommRows, self.angleOverlap, self.minAcc, self.maxAcc)
                #print "     FB COMMAND=",upperCommInput

                # For level 0 update the input space for level 0 and the display widget. 
                if lev==0:
                    inputSpace = self.angleInput  
                    inputSpace = np.vstack((inputSpace,upperCommInput))    
                    self.inputSpace = np.empty_like(inputSpace)
                    self.inputSpace[:] = inputSpace # Make a deep copy of the new input

                    

                # Add together the upper fb comm, lower levels output and command Space to create the total input for the current level.
                inTotalSpace = np.vstack((lowerLevelOutput,upperCommInput,newCommInput))
                newInput = np.empty_like(inTotalSpace)
                newInput[:] = inTotalSpace

                # Run the input through the HTM level.
                # This also increments the time for that level.
                self.HTMNetworkGrid.step(newInput,lev)
                

                # Save the old angle for each level
                self.oldAngle[lev]=self.angle
                
                # Save the command
                self.previousCommand = copy.deepcopy(self.command)

        
                    
        
        
#         # STEP 3 CREATE THE NEW COMMAND AND RUN IT THROUGH EACH LEVEL
#         ###############################################################
#         # The new input to the command space comes from the upper level and 
#         # wether the previous command brought the angle closer to 90 degrees.
#         print "STEP 3"
#         # Calculate the command space of the highest level first
#         for lev in xrange(self.numLevels):
#             # If the iteration is a power of 2 update the higher levels as well
#             twoPowLev = math.pow(2,2*lev)
#             if self.iteration%twoPowLev==0:
#                 print " LEVEL %s"%lev
#                 # If the angle is further from 90 degs then don't set the command as active. 
#                 # The command will not be incremented. 
#                 print "     New angle = %s Old angle = %s"%(self.angle,self.oldAngle[0])
#                 commSuccessful = self.HTMNetworkGrid.commandSuccessful(lev)
#                 if commSuccessful == True:
#                     print "     level %s WON"%lev
#                     newCommInput=invertPen.createInput(self.command[lev], self.width, self.angleInputHeight, self.angleOverlap, -self.maxAcc, self.maxAcc)
#                 else:
#                     print "     level %s LOST"%lev
#                     newCommInput=invertPen.createInput('none', self.width, self.angleInputHeight, self.angleOverlap, -self.maxAcc, self.maxAcc)
#                 # if (abs(90-self.angle) < abs(90-self.oldAngle[0])):
#                 #     newCommInput=invertPen.createInput(self.command[lev], self.width, self.angleInputHeight, self.angleOverlap, -self.maxAcc, self.maxAcc)
#                 #     print "     level %s WON"%lev
#                 # else:
#                 #     newCommInput=invertPen.createInput('none', self.width, self.angleInputHeight, self.angleOverlap, -self.maxAcc, self.maxAcc)
#                 #     print "     level %s LOST"%lev
                
#                 commandSpace = np.empty_like(newCommInput)
#                 commandSpace[:] = newCommInput
#                 # Run the new total input through the HTM

#                 # Save the old angle for each level
#                 self.oldAngle[lev]=self.angle
# ##                # Update self.input for the input display widget.
# ##                self.input[lev][self.commandRow:,:] = self.commandSpace   

        print " level commands=",self.command

        if updateViewer==True:
            # Set the input viewers array to self.input
            self.inputGrid.setInput(self.inputSpace)
            self.inputGrid.updateInput()
            # Update the columns and cells of the HTM viewer
            self.HTMNetworkGrid.updateHTMGrid()

        print "------------------------------------------"
    
    #def createNewInput(self):
    #    for i in range(self.inputGrid.input
        
    def mouseMoveEvent(self,event):
        print "Enter!" 
    
class HTMGui(QtGui.QMainWindow):
    
    def __init__(self):
        super(HTMGui, self).__init__()
        
        self.initUI()
        
    def initUI(self):               
        layout = QtGui.QHBoxLayout()
        HTMWidget=HTMNetwork()
        layout.addWidget(HTMWidget);
        self.statusBar().showMessage('Ready')
        
        newInputAction = QtGui.QAction(QtGui.QIcon('grid.png'), '&New Input', self)        
        newInputAction.setStatusTip('create a new input')
        #newInputAction.triggered.connect(HTMWidget.createNewInput())
        drawLevelAction = QtGui.QAction(QtGui.QIcon('grid.png'), '&Level', self)        
        drawLevelAction.setStatusTip('draw Level x')
        #drawLevelAction.triggered.connect(HTMWidget.drawLevel())
        
        self.toolbar = self.addToolBar('create a new input')
        self.toolbar.addAction(newInputAction)
        self.toolbar = self.addToolBar('draw a different Level')
        self.toolbar.addAction(drawLevelAction)
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        ViewMenu = menubar.addMenu('&View')
        fileMenu.addAction(newInputAction)
        ViewMenu.addAction(drawLevelAction)
        
        self.setGeometry(600, 100, 600, 750)
        self.setWindowTitle('HTM')
        self.setCentralWidget(HTMWidget);    
        self.show()

def main():   
    app = QtGui.QApplication(sys.argv)
    ex = HTMGui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


