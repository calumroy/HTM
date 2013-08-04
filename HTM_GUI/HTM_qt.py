#!/usr/bin/python

"""
HTM veiwer

This code draws the input and a HTM network in a grid using PyQt

author: Calum Meiklejohn
website: calumroy.com
last edited: June 2013
"""
import sys
sys.path.insert(0, '../')       #Add the parent diectory to the path to search for modules
import numpy as np
import random
import sys
from PyQt4 import QtGui, QtCore
import HTM_V12
import math

import HTM_lineInput

##class HTMInput(QtGui.QWidget):
##
##    def __init__(self,width,height):
##        super(HTMInput, self).__init__()
##        self.initUI(width,height)
##
##    def initUI(self,width,height):
##        self.size = 10
##        self.cols = width
##        self.rows = height
##        self.scale = 4
##        self.pos_x = 0
##        self.pos_y = 0
##        #sld = QtGui.QSlider(QtCore.Qt.Horizontal, self
##        self.setGeometry(300, 300, 350, 100)
##        self.setWindowTitle('Colors')
##        self.setGeometry(QtCore.QRect(0, 0, 400, 400))
##        self.inputArray = np.array([[0 for i in range(width)] for j in range(height)])
##        self.show()
##
##    def paintEvent(self, e):
##        qp = QtGui.QPainter()
##        qp.begin(self)
##        qp.scale(self.scale, self.scale)
##        qp.translate(self.pos_x,self.pos_y)
##        self.drawGrid(qp,self.rows,self.cols,self.size)
##        qp.end()
##
##    def drawGrid(self, qp, rows, cols, size):
##        color = QtGui.QColor(0, 0, 0)
##        color.setNamedColor('#d4d4d4')
##        qp.setPen(color)
##        #print "self.rows, self.cols = (%s,%s)"%(self.rows, self.cols)
##        #print "rows, cols = (%s,%s)" %(rows, cols)
##        for y in range(rows):
##                for x in range(cols):
##                        #print "x, y = (%s,%s)" %(x, y)
##                        #print "inputArray c,r = (%s,%s)"%(len(self.inputArray[0]),len(self.inputArray))
##                        value = self.inputArray[y][x]
##                        if value == 0:
##                                qp.setBrush(QtGui.QColor(200, 0, 0))
##                        else:
##                                qp.setBrush(QtGui.QColor(0, 200, 0))
##                        qp.drawRect( x*size,y*size, size, size)
##
##
##    def setInput(self,newInput):
##        self.cols = len(newInput[0])
##        self.rows = len(newInput)
##        #print "rows, cols = (%s,%s)" %(self.rows, self.cols)
##        self.inputArray = newInput

##class HTMGridViewer(QtGui.QWidget):
##    
##    def __init__(self,width,height):
##        super(HTMGridViewer, self).__init__()
##        self.initUI(width,height)
##        
##    def initUI(self,width,height):
##        self.size = 10
##        self.cols = width
##        self.rows = height
##        self.scale = 4
##        self.pos_x = 0
##        self.pos_y = 0
##        self.numCells = 3
##        self.showActiveCells = True
##        self.showPredictCells = False
##        self.showLearnCells = False
##        #sld = QtGui.QSlider(QtCore.Qt.Horizontal, self
##        self.setGeometry(300, 300, 350, 100)
##        self.setWindowTitle('Colors')
##        self.setGeometry(QtCore.QRect(0, 0, 400, 400))
##        self.columnActiveArray = np.array([[0 for i in range(width)] for j in range(height)])
##        self.cellsActiveArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
##        self.cellsPredictiveArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
##        self.cellsLearnArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
##        # Scale the size of the grid so the cells can be shown if there are too many cells
##        while (int (math.ceil(self.numCells ** 0.5))>self.size/2):
##            self.size = self.size *2
##        self.show()
##
##    def paintEvent(self, e):
##        qp = QtGui.QPainter()
##        qp.begin(self)
##        qp.scale(self.scale, self.scale)
##        qp.translate(self.pos_x,self.pos_y)
##        self.drawGrid(qp,self.rows,self.cols,self.size)
##        qp.end()
##        
##    def drawGrid(self, qp,rows, cols, size):
##        color = QtGui.QColor(0, 0, 0)
##        color.setNamedColor('#d4d4d4')
##        qp.setPen(color)
##        for y in range(rows):
##                for x in range(cols):
##                    v = self.columnActiveArray[y][x]
##                    if v == 0:
##                            qp.setBrush(QtGui.QColor(200, 0, 0))
##                    if v == 1:
##                            #print"PAINTING grid cell x,y=%s,%s"%(x,y)
##                            qp.setBrush(QtGui.QColor(0, 200, 0))
##                    qp.drawRect(x*size, y*size, size, size)
##                    self.drawCells(qp,self.numCells,x,y,size)
##
##    def drawCells(self,qp,numCells,pos_x,pos_y,size):
##        qp.setBrush(QtGui.QColor(0, 10, 100))
##        qp.setOpacity(0.2)
##        # Find the smallest number which when squared is larger than numCells
##        numSquares = int (math.ceil(numCells ** 0.5))
##        # Set the small sqaures to a size smaller than the large ones
##        squareSize = size/(1.5*numSquares)
##        # Count the cells that are drawn so we can identify them
##        count = 0
##        for i in range(numSquares):
##            for j in range(numSquares):
##                qp.setOpacity(0.05) # Make the non existent cells faint
##                if count < numCells:
##                    if self.showActiveCells==True:
##                        if self.cellsActiveArray[pos_y][pos_x][count] == 1:
##                            qp.setOpacity(1.0)
##                        else:
##                            qp.setOpacity(0.2)
##                    if self.showPredictCells==True:
##                        if self.cellsPredictiveArray[pos_y][pos_x][count] == 1:
##                            qp.setBrush(QtGui.QColor(50, 100, 10))
##                            qp.setOpacity(1.0)
##                        else:
##                            qp.setBrush(QtGui.QColor(0, 10, 100))
##                            qp.setOpacity(0.2)
##                    if self.showLearnCells==True:
##                        if self.cellsLearnArray[pos_y][pos_x][count] == 1:
##                            qp.setBrush(QtGui.QColor(100, 50, 10))
##                            qp.setOpacity(1.0)
##                        else:
##                            qp.setBrush(QtGui.QColor(0, 10, 100))
##                            qp.setOpacity(0.2)
##                count += 1
##                # Separate the small squares
##                x = pos_x*size + 0.5*squareSize+1.5*i*squareSize
##                y = pos_y*size + 0.5*squareSize+1.5*j*squareSize
##                qp.drawRect(x, y, squareSize, squareSize)
##        qp.setOpacity(1.0)

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
        
    def mousePressEvent(self,event):
        print"column pos_x,pos_y = %s,%s"%(self.pos_x,self.pos_y)



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
    
    def mousePressEvent(self,event):
        print"cell pos_x,pos_y,cell = %s,%s,%s"%(self.pos_x,self.pos_y,self.cell)


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
    
    def __init__(self,width,height):
        super(HTMGridViewer, self).__init__()
        self.initUI(width,height)
        
    def initUI(self,width,height):
        self.scene=QtGui.QGraphicsScene(self)
        self.scaleSize = 1
        self.setScene(self.scene)
        self.size = 10
        self.cols = width
        self.rows = height
        self.pos_x = 0
        self.pos_y = 0
        self.numCells = 3
        
        self.showActiveCells = True
        self.showPredictCells = False
        self.showLearnCells = False

        self.columnActiveArray = np.array([[0 for i in range(width)] for j in range(height)])
        self.cellsActiveArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
        self.cellsPredictiveArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
        self.cellsLearnArray = np.array([[[0 for k in range(self.numCells)] for i in range(width)] for j in range(height)])
        
        self.scaleGridSize()
        self.cellItems = []   # Stores all the cell items in the scene
        self.columnItems = [] # Stores all the column items in the scene
        
        self.drawGrid(self.rows,self.cols,self.size)
        self.show()
        
    def scaleGridSize(self):
        # Scale the size of the grid so the cells can be shown if there are too many cells
        while (int (math.ceil(self.numCells ** 0.5))>self.size/2):
            self.size = self.size *2
            
    def drawGrid(self,rows, cols, size):
        pen   = QtGui.QPen(QtGui.QColor(QtCore.Qt.black))
        brush = QtGui.QBrush(pen.color().darker(150))
        for x in range(rows):
                for y in range(cols):
                    v = self.columnActiveArray[y][x]
                    if v == 0:
                            brush.setColor(QtCore.Qt.red)
                    if v == 1:
                            brush.setColor(QtCore.Qt.green)
                    # Create a column item and add it to a list so we can iterate through them to update
                    columnItem = HTMColumn(x,y,size,pen,brush)
                    columnItem.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)  
                    #columnItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)                  
                    self.columnItems.append(columnItem) 
                    self.scene.addItem(columnItem)
                    self.drawCells(self.numCells,x,y,size)

    def drawCells(self,numCells,pos_x,pos_y,size):
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
                    #cellItem = self.scene.addRect(x,y,squareSize,squareSize, pen, brush)
                    cellItem = HTMCell(pos_x,pos_y,count,x,y,squareSize,pen,brush)
                    #cellItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
                    self.cellItems.append(cellItem) #Add the cells to a list so we can iterate through them to update
                    self.scene.addItem(cellItem)    # Add the cells to the scene
                # Increase the count to keep track of how many cells have been created
                count += 1
                
    
    def updateColumns(self):
        for i in range(len(self.columnItems)):
            brush = QtGui.QBrush(QtCore.Qt.green)
            brush.setStyle(QtCore.Qt.SolidPattern)
            pos_x=self.columnItems[i].pos_x
            pos_y=self.columnItems[i].pos_y
            value = self.columnActiveArray[pos_y][pos_x]
            if value == 0:
                    brush.setColor(QtCore.Qt.red)
                    self.columnItems[i].setBrush(brush)
            if value == 1:
                    brush.setColor(QtCore.Qt.green)
                    self.columnItems[i].setBrush(brush)
        self.updateCells()
                    
    def updateCells(self):
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
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
                if self.cellsActiveArray[pos_y][pos_x][cell] == 1:
                    brush.setColor(blue);
                else:
                    brush.setColor(transpBlue);
            if self.showPredictCells==True:
                if self.cellsPredictiveArray[pos_y][pos_x][cell] == 1:
                    brush.setColor(green);
                else:
                    brush.setColor(transpBlue);
            if self.showLearnCells==True:
                if self.cellsLearnArray[pos_y][pos_x][cell] == 1:
                    brush.setColor(darkGreen);
                else:
                    brush.setColor(transpBlue);
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)
                
    
    def scaleScene(self,scaleSize):
        self.scale(scaleSize, scaleSize)
        
class HTMNetwork(QtGui.QWidget):

    def __init__(self):
        super(HTMNetwork, self).__init__()
        self.initUI()

    def initUI(self):
        self.iteration = 0
        width=20
        height=20
        self.scaleFactor=0.2    # How much to scale the grids by
        self.input = self.setInput(width,height)
        self.patternsArray = HTM_lineInput.createPatternArray(3,width,height,4,0,30)
        self.htm = HTM_V12.HTM(1,self.input,width,height)
        self.HTMNetworkGrid = HTMGridViewer(width,height)
        self.inputGrid = HTMInput(width,height)
        self.setHTMViewer(self.HTMNetworkGrid)
        self.make_frame()
        self.make_buttons()
        self.setGeometry(600, 600, 700, 500)
        #self.setWindowTitle('Main window')
        self.show()

    def setHTMViewer(self,HTMViewer):
        HTMViewer.cols = self.htm.HTMLayerArray[0].width
        HTMViewer.rows = self.htm.HTMLayerArray[0].height
        for i in range(len(self.htm.HTMLayerArray[0].columns)):
            for j in range(len(self.htm.HTMLayerArray[0].columns[i])):
                    # Show the active columns
                    if self.htm.HTMLayerArray[0].columns[i][j].activeState == True:
                        HTMViewer.columnActiveArray[i][j] = 1
                        #print"PAINTING grid cell x,y=%s,%s"%(j,i)
                    else:
                        HTMViewer.columnActiveArray[i][j] = 0
                    # Show the active cells
                    for c in range(self.htm.HTMLayerArray[0].cellsPerColumn):
                        # Convert to an int the array is in floats
                        if int(self.htm.HTMLayerArray[0].columns[i][j].activeStateArray[c]) == self.iteration:
                            HTMViewer.cellsActiveArray[i][j][c] = 1
                        else:
                            HTMViewer.cellsActiveArray[i][j][c] = 0
                            #if int(self.htm.HTMLayerArray[0].columns[i][j].activeStateArray[c])!=0:
                            #    print "\n HTMViewer iteration=%s activeStateArray=%s for x,y,i = %s,%s,%s"%(self.iteration,self.htm.HTMLayerArray[0].columns[i][j].activeStateArray[c],j,i,c)
                    # Show the predictive cells
                    for c in range(self.htm.HTMLayerArray[0].cellsPerColumn):
                        if int(self.htm.HTMLayerArray[0].columns[i][j].predictiveStateArray[c]) == self.iteration:
                            HTMViewer.cellsPredictiveArray[i][j][c] = 1
                        else:
                            HTMViewer.cellsPredictiveArray[i][j][c] = 0
                    # Show the learning cells
                    for c in range(self.htm.HTMLayerArray[0].cellsPerColumn):
                        if int(self.htm.HTMLayerArray[0].columns[i][j].learnStateArray[c]) == self.iteration:
                            HTMViewer.cellsLearnArray[i][j][c] = 1
                        else:
                            HTMViewer.cellsLearnArray[i][j][c] = 0

    def setInput(self,width,height):
        input = np.array([[0 for i in range(width)] for j in range(height)])
        return input

    def make_buttons(self):
        btn1 = QtGui.QPushButton("HTM +", self)
        btn1.clicked.connect(self.HTMzoomIn)
        btn2 = QtGui.QPushButton("HTM -", self)
        btn2.clicked.connect(self.HTMzoomOut)
        btn3 = QtGui.QPushButton("->", self)
        btn3.clicked.connect(self.moveLeft)
        btn4 = QtGui.QPushButton("<-", self)
        btn4.clicked.connect(self.moveRight)
        btn5 = QtGui.QPushButton("step", self)
        btn5.clicked.connect(self.step)
        btn6 = QtGui.QPushButton("In +", self)
        btn6.clicked.connect(self.inputZoomIn)
        btn7 = QtGui.QPushButton("IN -", self)
        btn7.clicked.connect(self.inputZoomOut)
        btn8 = QtGui.QPushButton("Active Cells", self)
        btn8.clicked.connect(self.showActiveCells)
        btn9 = QtGui.QPushButton("Predict Cells", self)
        btn9.clicked.connect(self.showPredictCells)
        btn10 = QtGui.QPushButton("Learn Cells", self)
        btn10.clicked.connect(self.showLearnCells)
        btn1.move(315,90)
        btn2.move(415,90)
        btn3.move(315,30)
        btn4.move(415,30)
        btn5.move(15, 30)
        btn6.move(15, 90)
        btn7.move(115, 90)
        btn8.move(215, 30)
        btn9.move(215, 60)
        btn10.move(215, 90)
        
    def showActiveCells(self):
        # Toggle between showing the active cells or not
        if self.HTMNetworkGrid.showActiveCells==True:
            self.HTMNetworkGrid.showActiveCells=False
        else:
            self.HTMNetworkGrid.showActiveCells=True
            # Don't show the learning or predictive cells
            self.HTMNetworkGrid.showPredictCells=False
            self.HTMNetworkGrid.showLearnCells=False
            
        self.HTMNetworkGrid.updateColumns()
    def showPredictCells(self):
        # Toggle between showing the predicting cells or not
        if self.HTMNetworkGrid.showPredictCells==True:
            self.HTMNetworkGrid.showPredictCells=False
        else:
            self.HTMNetworkGrid.showPredictCells=True
            # Don't show the learning or active cells
            self.HTMNetworkGrid.showActiveCells=False
            self.HTMNetworkGrid.showLearnCells=False
        self.HTMNetworkGrid.updateColumns()
    def showLearnCells(self):
        # Toggle between showing the learning cells or not
        if self.HTMNetworkGrid.showLearnCells==True:
            self.HTMNetworkGrid.showLearnCells=False
        else:
            self.HTMNetworkGrid.showLearnCells=True
            # Don't show the learning or active cells
            self.HTMNetworkGrid.showActiveCells=False
            self.HTMNetworkGrid.showPredictCells=False
        self.HTMNetworkGrid.updateColumns()
    def HTMzoomIn(self):
        #self.HTMNetworkGrid.scale = self.HTMNetworkGrid.scale*1.2
        #self.HTMNetworkGrid.update()
        self.HTMNetworkGrid.scaleScene(1+self.scaleFactor)
    def HTMzoomOut(self):
        #self.HTMNetworkGrid.scale = self.HTMNetworkGrid.scale*0.8
        #self.HTMNetworkGrid.update()
        self.HTMNetworkGrid.scaleScene(1-self.scaleFactor)
    def moveLeft(self):
        self.HTMNetworkGrid.pos_x += 2
        self.HTMNetworkGrid.update()
    def moveRight(self):
        self.HTMNetworkGrid.pos_x -= 2
        self.HTMNetworkGrid.update()
    def inputZoomIn(self):
        self.inputGrid.scaleScene(1+self.scaleFactor)
    def inputZoomOut(self):
        self.inputGrid.scaleScene(1-self.scaleFactor)

    def make_frame(self):
        frame1 = QtGui.QFrame(self)
        frame1.setLineWidth(3)
        frame1.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        frame2 = QtGui.QFrame(self)
        frame2.setLineWidth(3)
        frame2.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        frame3 = QtGui.QFrame(self)
        frame3.setLineWidth(3)
        frame3.setFrameStyle(QtGui.QFrame.Box|QtGui.QFrame.Sunken)
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)
        # addWidget(QWidget, row, column, rowSpan, columnSpan)
        grid.addWidget(self.HTMNetworkGrid,2,3,2,2)
        grid.addWidget(self.inputGrid,2,1,2,2)
        grid.addWidget(frame1, 1, 1, 1, 4)
        grid.addWidget(frame2, 2, 1, 2, 2)
        grid.addWidget(frame3, 2, 3, 2, 2)
        self.setLayout(grid)

    def step(self):
        self.iteration += 1
        # Temporary code to create a test input pattern
        # Make sure the input is larger than this test input
        for k in range(len(self.input)):
            for l in range(len(self.input[k])):
                #self.input[k][l] = 0
                # Add some noise
                some_number = round(random.uniform(0,10))
                if some_number>10:
                    self.input[k][l] = 1
        if self.iteration % 3 == 0:
            print "\n pattern1"
            self.input=self.patternsArray[0]
        else:
            if self.iteration%3==1:
                print "\n pattern2"
                self.input=self.patternsArray[1]
        if self.iteration%3==2:
            print "\n pattern3"
            self.input=self.patternsArray[2]
        # Put the new input through the htm
        self.htm.spatial_temporal(self.input)
        # Set the input viewers array to self.input
        self.inputGrid.setInput(self.input)
        self.inputGrid.updateInput()
        # Set the HTM viewers array to the new state of the htm network
        self.setHTMViewer(self.HTMNetworkGrid)
        self.HTMNetworkGrid.updateColumns()
    
    #def createNewInput(self):
    #    for i in range(self.inputGrid.input
        
    #def mouseMoveEvent(self,event):
    #    print "Enter!" 
    
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

        self.toolbar = self.addToolBar('create a new input')
        self.toolbar.addAction(newInputAction)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newInputAction)
        
        self.setGeometry(300, 330, 800, 600)
        self.setWindowTitle('HTM')
        self.setCentralWidget(HTMWidget);    
        self.show()

def main():   
    app = QtGui.QApplication(sys.argv)
    ex = HTMGui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


