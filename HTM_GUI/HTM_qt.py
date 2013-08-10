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
from PyQt4.QtCore import QObject, pyqtSlot
import HTM_V12
import math

import HTM_lineInput

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
                self.close()

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
        # For the popup segment selection box
        self.segmentSelect = None
        self.selectedItem = None
        
        # Create HTM network with an empty input
        self.iteration = 0
        input = np.array([[0 for i in range(width)] for j in range(height)])
        self.htm = HTM_V12.HTM(1,input,width,height)
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
        for x in range(rows):
                for y in range(cols):
                    value = self.htm.HTMLayerArray[0].columns[y][x].activeState
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
                
    def drawSingleCell(self,pos_x,pos_y,cell,segment):
        # Draw the cells connected to the selected segment
        print"pos_x,pos_y,cell,seg = %s,%s,%s,%s"%(pos_x,pos_y,cell,segment)
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        # Go through each cell is it is in the synapse list draw it otherwise don't
        for i in range(len(self.cellItems)):
            #print"HEYEHEYEHHYE"
            cell_pos_x=self.cellItems[i].pos_x
            cell_pos_y=self.cellItems[i].pos_y
            cell_cell=self.cellItems[i].cell
            brush = QtGui.QBrush(transpBlue)   # Have to create a brush with a color
            # Check each synapse and draw the connected cells
            for syn in self.htm.HTMLayerArray[0].columns[pos_y][pos_x].cells[cell].segments[segment].synapses:
                if syn.pos_x==cell_pos_x and syn.pos_y==cell_pos_y and syn.cell==cell_cell:
                    brush.setColor(blue);
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)
                
    
    def updateHTMGrid(self):
        for i in range(len(self.columnItems)):
            brush = QtGui.QBrush(QtCore.Qt.green)
            brush.setStyle(QtCore.Qt.SolidPattern)
            pos_x=self.columnItems[i].pos_x
            pos_y=self.columnItems[i].pos_y
            value = self.htm.HTMLayerArray[0].columns[pos_y][pos_x].activeState
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
                if int(self.htm.HTMLayerArray[0].columns[pos_y][pos_x].activeStateArray[cell]) == self.iteration:
                    brush.setColor(blue);
                else:
                    brush.setColor(transpBlue);
            if self.showPredictCells==True:
                if int(self.htm.HTMLayerArray[0].columns[pos_y][pos_x].predictiveStateArray[cell]) == self.iteration:
                    brush.setColor(green);
                else:
                    brush.setColor(transpBlue);
            if self.showLearnCells==True:
                if int(self.htm.HTMLayerArray[0].columns[pos_y][pos_x].learnStateArray[cell]) == self.iteration:
                    brush.setColor(darkGreen);
                else:
                    brush.setColor(transpBlue);
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)
                
    
    def scaleScene(self,scaleSize):
        self.scale(scaleSize, scaleSize)
        
    def mousePressEvent(self,event):
        item=self.itemAt(event.x(),event.y())
        if item.__class__.__name__ == "HTMCell":
            print"cell"
            print "pos_x,pos_y,cell = %s,%s,%s"%(item.pos_x,item.pos_y,item.cell)
            numSegments = len(self.htm.HTMLayerArray[0].columns[item.pos_y][item.pos_x].cells[item.cell].segments)
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
        
        
    def step(self,input,timeStep):
        self.htm.spatial_temporal(input)
        self.iteration = timeStep
        
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
        self.HTMNetworkGrid = HTMGridViewer(width,height)
        self.inputGrid = HTMInput(width,height)
        self.make_frame()
        self.make_buttons()
        self.setGeometry(600, 600, 700, 500)
        #self.setWindowTitle('Main window')
        self.show()

    def setInput(self,width,height):
        input = np.array([[0 for i in range(width)] for j in range(height)])
        return input
    

    def make_buttons(self):
        btn1 = QtGui.QPushButton("HTM +", self)
        btn1.clicked.connect(self.HTMzoomIn)
        btn2 = QtGui.QPushButton("HTM -", self)
        btn2.clicked.connect(self.HTMzoomOut)
        btn3 = QtGui.QPushButton("All HTM", self)
        btn3.clicked.connect(self.showAllHTM)

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
        btn3.move(115,30)
        
        btn5.move(15, 30)
        btn6.move(15, 90)
        btn7.move(115, 90)
        btn8.move(215, 30)
        btn9.move(215, 60)
        btn10.move(215, 90)
    
    def showAllHTM(self):
        # Draw the entire HTM netwrok. This is used if previously just a 
        # single cells segment connection was being shown
        self.HTMNetworkGrid.showAllHTM = True
        self.HTMNetworkGrid.updateHTMGrid()
        
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
        #self.htm.spatial_temporal(self.input)
        #self.HTMViewer.htm.spatial_temporal(self.input)
        self.HTMNetworkGrid.step(self.input,self.iteration)
        # Set the input viewers array to self.input
        self.inputGrid.setInput(self.input)
        self.inputGrid.updateInput()
        # Update the columns and cells of the HTM viewer
        self.HTMNetworkGrid.updateHTMGrid()
    
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


