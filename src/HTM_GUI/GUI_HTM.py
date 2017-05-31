#!/usr/bin/python

"""
HTM GUI
author: Calum Meiklejohn
website: calumroy.com

This code draws the input and a HTM network in a grid using PyQt
It creates a simple input and output Grid of squares representing the input
space and HTM columns.

    Red squares are inactive inputs or columns.

    Green squares are active input or columns.

    Smaller Blue squares are active cells in the columns.

    Smaller black squares are predicting cells in the columns.

    Smaller Green squares are cells in the learing state in the columns.

    Clicking on the columns displays the potential and connected proximal
    synapses for that column. White or dark greenshow the connected synapses.

    Clicking on the cells shows the connected distal synpases for that cell by
    highlightling the other cells that synapses connect to from a selected segment.

    You Can "Mark" the current state of a HTM layer by clicking the mark button.
    This creates a copy of the current layer and displays it in a new window.

    Some shortcut keys exist
        Space: Run one timestep
        1,2,3: display the active, predictive and learning cells.

"""
import sys
#, Add the parent diectory to the path to search for modules
sys.path.insert(0, './')
import numpy as np
import random

from PyQt4 import QtGui, QtCore
#from PyQt4.QtCore import QObject
import HTM_network as HTM_V
from reinforcement_learning import Thalamus
import math
import copy

import decimal



class layerPopup(QtGui.QWidget):
    # A popup menu to slect a certain layer to display in the HTM
    # Create a signal to tell the network which layer was selected
    levelSelectedSignal = QtCore.pyqtSignal(int)

    def __init__(self, numLayers):
        QtGui.QWidget.__init__(self)
        self.numLayers = numLayers
        layout = QtGui.QVBoxLayout()
        self.checks = []
        for i in range(self.numLayers):
            c = QtGui.QCheckBox(" Layer %s " % i)
            c.stateChanged.connect(self.levelSelected)
            layout.addWidget(c)
            self.checks.append(c)
        self.setLayout(layout)

    def levelSelected(self, i):
        # Check each check box to find out which one was selected
        for i in range(len(self.checks)):
            if self.checks[i].isChecked() is True:
                self.levelSelectedSignal.emit(i)
                #print"levelSelectedSignal sent"

    def paintEvent(self, event):
        dc = QtGui.QPainter(self)


class levelPopup(QtGui.QWidget):
    # A popup menu to slect a certain level to display in the HTM

    # Create a signal to tell the network which level was selected
    levelSelectedSignal = QtCore.pyqtSignal(int)

    def __init__(self, numLevels):
        QtGui.QWidget.__init__(self)
        self.numLevels = numLevels
        layout = QtGui.QVBoxLayout()
        self.checks = []
        for i in range(self.numLevels):
            c = QtGui.QCheckBox("Level %s" % i)
            c.stateChanged.connect(self.levelSelected)
            layout.addWidget(c)
            self.checks.append(c)
        self.setLayout(layout)

    def levelSelected(self, i):
        # Check each check box to find out which one was selected
        for i in range(len(self.checks)):
            if self.checks[i].isChecked() is True:
                self.levelSelectedSignal.emit(i)
                #print"levelSelectedSignal sent"

    def paintEvent(self, event):
        dc = QtGui.QPainter(self)


class popup(QtGui.QWidget):
    # Create a signal to tell the scene which segment was selected
    segmentSelectedSignal = QtCore.pyqtSignal(int)

    def __init__(self, x, y, cell, numSegments):
        QtGui.QWidget.__init__(self)
        self.pos_x = x
        self.pos_y = y
        self.cell = cell
        self.numSegments = numSegments
        layout = QtGui.QVBoxLayout()
        self.checks = []
        for i in range(self.numSegments):
            c = QtGui.QCheckBox("segment %s" % i)
            c.stateChanged.connect(self.segmentSelected)
            layout.addWidget(c)
            self.checks.append(c)
        self.setLayout(layout)

    def segmentSelected(self, i):
        # Check each check box to find out which one was selected
        for i in range(len(self.checks)):
            if self.checks[i].isChecked() is True:
                self.segmentSelectedSignal.emit(i)
                #print"segmentSelectedSignal sent"
                #self.close()

    def paintEvent(self, event):
        dc = QtGui.QPainter(self)


class HTMInfo(QtGui.QGraphicsItem):
    # A class used to show which level and layer of the HTM network is displayed.
    def __init__(self, x, y, squareSize, pen, brush):
        super(HTMInfo, self).__init__()
        self.initUI(x, y, squareSize, pen, brush)

    def initUI(self, x, y, squareSize, pen, brush):
        self.info = "Hello"
        self.setPos(x, y)
        self.width = squareSize
        self.height = squareSize
        #self.setScale(squareSize)
        #self.setPen(pen)
        #self.setBrush(brush)
        #self.setRect(0, 0, squareSize, squareSize)
        #self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)

    def updateInfo(self, info):
        self.info = info
        self.update()

    def paint(self, painter, option, widget):
        #qp = QtGui.QPainter()
        #qp.drawText(event, qp)
        painter.drawRect(0, 0, self.width, self.height)
        painter.drawText(0, 0, self.width, self.height, QtCore.Qt.TextWordWrap, self.info)

    def boundingRect(self):
        #return QtCore.QRectF(self.x(), self.y(), self.width, self.height)
        penWidth = 1
        return QtCore.QRectF(-penWidth, - penWidth,
                             self.width + penWidth, self.height + penWidth)

    def drawText(self, event, qp):
        qp.setPen(QtGui.QColor(255, 34, 3))
        qp.setFont(QtGui.QFont(self.info, 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignCenter, self.text)


class HTMColumn(QtGui.QGraphicsRectItem):
    # A class used to display the column of a HTM network
    def __init__(self, HTM_x, HTM_y, squareSize, pen, brush):
        super(HTMColumn, self).__init__()
        self.initUI(HTM_x, HTM_y, squareSize, pen, brush)

    def initUI(self, HTM_x, HTM_y, squareSize, pen, brush):
        self.pos_x = HTM_x  # The x position in the HTM grid
        self.pos_y = HTM_y  # The y position in the HTM grid
        self.setPos(HTM_x*squareSize, HTM_y*squareSize)
        self.setRect(0, 0, squareSize, squareSize)
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

    #def mousePressEvent(self,event):
    #    print"column pos_x,pos_y = %s,%s"%(self.pos_x,self.pos_y)


class HTMCell(QtGui.QGraphicsRectItem):
    # A class used to display the cell of a HTM network
    def __init__(self, HTM_x, HTM_y, cell, x, y, squareSize, pen, brush):
        super(HTMCell, self).__init__()
        self.initUI(HTM_x, HTM_y, cell, x, y, squareSize, pen, brush)

    def initUI(self, HTM_x, HTM_y, cell, x, y, squareSize, pen, brush):
        self.pos_x = HTM_x  # The x position in the HTM grid
        self.pos_y = HTM_y  # The y position in the HTM grid
        self.cell = cell  # The cell number oin the HTM grid
        self.setPos(x, y)
        self.setRect(0, 0, squareSize, squareSize)
        self.setPen(pen)
        self.setBrush(brush)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

    #def mousePressEvent(self,event):
    #    print"cell pos_x,pos_y,cell = %s,%s,%s"%(self.pos_x,self.pos_y,self.cell)


class HTMInput(QtGui.QGraphicsView):
    # A class used to dispaly the input to a HTM
    def __init__(self, htm, HTMGridViewer):
        super(HTMInput, self).__init__()

        self.initUI(htm, HTMGridViewer)

    def initUI(self, htm, HTMGridViewer):
        self.scene = QtGui.QGraphicsScene(self)
        self.scaleSize = 1
        self.setScene(self.scene)
        self.size = 20
        self.pos_x = 0
        self.pos_y = 0
        # Store the htm so it can be referenced in other functions.
        self.htm = htm

        self.level = 0  # Draw this level (Region) of the HTMNetwork
        self.layer = 0  # Draw this HTM layer in the level.
        self.cols = 0
        self.rows = 0
        self.columnItems = []  # Stores all the column items in the scene

        # Keep track of the right mouse button to move the view
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self._mousePressed = False
        self._dragPos = None

        # Connect up the slots and signals
        self.connectSlots(HTMGridViewer)

        self.scaleScene(0.8)
        self.drawGrid(self.size)
        self.show()
        # Fit the scene into the view
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def connectSlots(self, HTMGridVeiwer):
        # Create and connect slots between the HTMGridVeiwer and this HTMInput Veiwer
        # create a slot so if a column is selected in the HTMGridViewer this
        # HTMInput can tell which column it was and highlight the
        # grid input squares which are connected to that columns connected synapses.
        HTMGridVeiwer.selectedColumn.connect(self.drawColumnInputs)

    def contains(self, list, filter):
        # Helper function for quickly searching a list of column objects.
        for x in list:
            if filter(x):
                return True
        return False

    def drawColumnInputs(self, pos_x, pos_y):
        self.updateGrid()
        #print "Selected pos_x = %s pos_y = %s" % (pos_x, pos_y)
        currentLayer = self.htm.regionArray[self.level].layerArray[self.layer]
        column = currentLayer.columns[pos_y][pos_x]
        columnsOverlap = currentLayer.getColumnsOverlap(column)
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        potColumnsOverlap = currentLayer.getPotentialOverlaps(column)
        columnsMinOverlap = currentLayer.getColumnsMinOverlap()
        print "      overlap = %s\n      potColumnsOverlap = %r\n      minOverlap = %s" % (columnsOverlap, potColumnsOverlap, columnsMinOverlap)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        darkerGreen = QtGui.QColor(0x1A, 0x39, 0x26, 0xFF)
        lightRed = QtGui.QColor(0xFF, 0x60, 0x60, 0xFF)
        lightGreen = QtGui.QColor(0, 0xFF, 0, 0x80)

        # Get the selected columns synpases from the htm layer.
        selectedColumn = currentLayer.columns[pos_y][pos_x]
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        colSynList = currentLayer.getConnectedSynapses(selectedColumn)
        # Get the potential list of synapses for the selected column
        potColSynList = currentLayer.getPotColSynapses(selectedColumn)
        print "        Number of potential synpases = %s" % len(potColSynList) 
    
        # Color the inputs which have potential column synapses connected to
        # them for the selected column slightly lighter then the other inputs.
        for syn in potColSynList:
            # Find the columnItems that have the same x and y pos
            # as the synpase ends in the colSynList.
            # The self.columnItems is ordered but not in the same way as
            # The htm input grid.
            # Check that the potential synpases isn't conected to padding on
            # the input grid. If so don't draw the input at -1 numpy arrays allow negative indexes.
            if (syn.pos_y != -1 and syn.pos_x != -1):
                syn_index = syn.pos_y+syn.pos_x*self.rows
                col = self.columnItems[syn_index]
                value = self.htm.regionArray[self.level].layerArray[self.layer].Input[col.pos_y][col.pos_x]

                color = QtGui.QColor(0xFF, 0, 0, 0xFF)
                brush = QtGui.QBrush(QtCore.Qt.red)

                if (value == 0):
                    color = lightRed
                elif (value == 1):
                    color = lightGreen

                brush.setColor(color)
                col.setBrush(brush)

        print "        Connected Col Synapse Permanence values > %s (x,y, perm):"
        # Color the inputs which have "connected" column synapses connected to
        # them for the selected column differently to the other inputs.
        for syn in colSynList:
            # Find the columnItems that have the same x and y pos
            # as the synpase ends in the colSynList.
            # The self.columnItems is ordered but not in the same way as
            # The htm input grid.
            syn_index = syn.pos_y+syn.pos_x*self.rows
            col = self.columnItems[syn_index]
            value = self.htm.regionArray[self.level].layerArray[self.layer].Input[col.pos_y][col.pos_x]

            print "        (%s, %s, %s)" %(syn.pos_x, syn.pos_y, syn.permanence)
            color = QtGui.QColor(0xFF, 0, 0, 0xFF)
            brush = QtGui.QBrush(QtCore.Qt.red)

            if (value == 0):
                color = transpBlue
            elif (value == 1):
                color = darkGreen

            brush.setColor(color)
            col.setBrush(brush)

    def scaleScene(self, scaleSize):
        self.scaleSize = self.scaleSize*scaleSize
        self.scale(scaleSize, scaleSize)

    def resizeView(self):
        # Resize the view. This function is called by some qt signals.
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def drawGrid(self, size):
        # Used to initialise the graphics scene with the input grid
        # Also used to draw a new layers input for the HTM since different layers cn have different sized inputs.
        # Remove the items from the scene
        for item in range(len(self.columnItems)):
            self.scene.removeItem(self.columnItems[item])
        # Clear the cellItems and columnItems arrays
        self.cellItems = []   # Stores all the cell items in the scene
        self.columnItems = []  # Stores all the column items in the scene
        self.rows = len(self.htm.regionArray[self.level].layerArray[self.layer].Input)
        self.cols = len(self.htm.regionArray[self.level].layerArray[self.layer].Input[0])
        pen = QtGui.QPen(QtGui.QColor(QtCore.Qt.black))
        brush = QtGui.QBrush(QtCore.Qt.red)
        print "Input rows = %s Input cols = %s" % (self.rows, self.cols)
        #print self.htm.regionArray[self.level].layerArray[self.layer].Input
        for x in range(self.cols):
                for y in range(self.rows):
                    #print "x = %s y = %s" % (x, y)
                    value = self.htm.regionArray[self.level].layerArray[self.layer].Input[y][x]
                    if value is False:
                            brush.setColor(QtCore.Qt.red)
                    if value is True:
                            brush.setColor(QtCore.Qt.green)
                    # Create a column item and add it to a list so we can iterate through them to update
                    columnItem = HTMColumn(x, y, size, pen, brush)
                    columnItem.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
                    #columnItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
                    self.columnItems.append(columnItem)
                    self.scene.addItem(columnItem)
        # Set the scene rectangle. This is an invisible rectangle that
        # defines where all the colums lie in the QGraphics scene.
        # It is used to scale the scene so everything fits in the view.
        self.scene.setSceneRect(0, 0, self.cols*size, self.rows*size)

    def updateGrid(self):
        for i in range(len(self.columnItems)):
            brush = QtGui.QBrush(QtCore.Qt.green)
            brush.setStyle(QtCore.Qt.SolidPattern)
            pos_x = self.columnItems[i].pos_x
            pos_y = self.columnItems[i].pos_y
            value = self.htm.regionArray[self.level].layerArray[self.layer].Input[pos_y][pos_x]
            if value == 0:
                    brush.setColor(QtCore.Qt.red)
                    self.columnItems[i].setBrush(brush)
            if value == 1:
                    brush.setColor(QtCore.Qt.green)
                    self.columnItems[i].setBrush(brush)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self._mousePressed = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._dragPos = event.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._mousePressed:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        else:
            super(HTMInput, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self.setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
            #self._mousePressed = False
        #if event.button() == QtCore.Qt.RightButton:
        #        super(HTMInput, self).mouseReleaseEvent(event)
        super(HTMInput, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Autoscale the scene to fit in the view
        # Get the position of the top most column item and
        # the bottom most left item.
        self.resizeView()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control and not self._mousePressed:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        else:
            super(HTMInput, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            if not self._mousePressed:
                self.setCursor(QtCore.Qt.ArrowCursor)

    def wheelEvent(self, event):
        factor = 1.2
        if event.delta() < 0:
            factor = 1.0 / factor
        self.scaleScene(factor)


class HTMGridViewer(QtGui.QGraphicsView):
    # Create a signal used by the HTMInput veiwer to tell it which
    # column was selected by the user.
    selectedColumn = QtCore.pyqtSignal(int, int)

    def __init__(self, htm):
        super(HTMGridViewer, self).__init__()
        self.initUI(htm)

    def initUI(self, htm):
        self.scene = QtGui.QGraphicsScene(self)
        self.scaleSize = 1
        self.setScene(self.scene)
        self.size = 20  # Size of the drawn cells
        self.numCells = htm.cellsPerColumn  # The number of cells in a column.
        self.level = 0  # Draw this level (Region) of the HTMNetwork
        self.layer = 0  # Draw this HTM layer in the level.
        # For the popup segment selection box
        self.segmentSelect = None
        self.selectedItem = None
        # Keep track of the right mouse button to move the view
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self._mousePressed = False
        self.dragView = False
        self.dragInfo = False
        self._dragPos = None

        # Store the htm so it can be referenced in other functions.
        self.htm = htm

        self.showAllHTM = True  # A flag to indicate to draw all the cells and column states
        self.showActiveCells = True
        self.showPredictCells = False
        self.showLearnCells = False

        self.scaleGridSize()
        self.cellItems = []   # Stores all the cell items in the scene
        self.columnItems = []  # Stores all the column items in the scene
        self.infoItem = None    # Stores the info item in the scene

        self.scaleScene(1.5)
        self.drawGrid(self.size)
        self.drawInfo()
        self.show()

        # Fit the scene into the view
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def selectedSegmentIndex(self, index):
        #print"Selected item pos_x,pos_y,cell,segment%s,%s,%s,%s"%(self.selectedItem.pos_x,self.selectedItem.pos_y,self.selectedItem.cell,index)
        self.drawSingleCell(self.selectedItem.pos_x, self.selectedItem.pos_y, self.selectedItem.cell, index)

    def scaleGridSize(self):
        # Scale the size of the grid so the cells can be shown if there are too many cells
        while (int(math.ceil(self.numCells ** 0.5)) > self.size/2):
            self.size = self.size*2

    def drawInfo(self):
        # Create an info item which displayes info about the HTM network
        transpRed = QtGui.QColor(0x00, 0, 0xFF, 0xA0)
        pen = QtGui.QPen(QtGui.QColor(transpRed))
        brush = QtGui.QBrush(pen.color().darker(150))
        self.infoItem = HTMInfo(-4*self.size, 0, 4*self.size, pen, brush)
        self.updateInfo()
        #self.infoItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.scene.addItem(self.infoItem)

    def updateInfo(self):
        if self.infoItem is not None:
            timeStep = self.htm.regionArray[self.level].layerArray[self.layer].timeStep
            self.infoItem.updateInfo("""layer = %s level = %s time = %s"""
                                     % (self.layer,
                                        self.level,
                                        timeStep))

    def drawGrid(self, size):
        # Used to initialise the graphics scene with columns and cells
        # Also used to draw a new layer of the HTM since different layers cn have different sized HTM grids.
        # Remove the items from the scene
        for item in range(len(self.cellItems)):
            self.scene.removeItem(self.cellItems[item])
        for item in range(len(self.columnItems)):
            self.scene.removeItem(self.columnItems[item])
        # Clear the cellItems and columnItems arrays
        self.cellItems = []   # Stores all the cell items in the scene
        self.columnItems = []  # Stores all the column items in the scene
        layer = self.htm.regionArray[self.level].layerArray[self.layer]
        rows = layer.height
        cols = layer.width
        pen = QtGui.QPen(QtGui.QColor(QtCore.Qt.black))
        brush = QtGui.QBrush(pen.color().darker(150))
        for x in range(cols):
                for y in range(rows):
                    # Check if the column at position x, y is active now
                    value = layer.columnActiveStateNow(x, y)
                    if value is False:
                            brush.setColor(QtCore.Qt.red)
                    if value is True:
                            brush.setColor(QtCore.Qt.green)
                    # Create a column item and add it to a list so we can iterate through them to update
                    columnItem = HTMColumn(x, y, size, pen, brush)
                    columnItem.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
                    #columnItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
                    self.columnItems.append(columnItem)
                    self.scene.addItem(columnItem)
                    self.drawCells(self.numCells, x, y, size)
        # Update the info
        self.updateInfo()
        # Set the scene rectangle. This is an invisible rectangle that
        # defines where all the colums lie in the QGraphics scene.
        # It is used to scale the scene so everything fits in the view.
        self.scene.setSceneRect(0, 0, cols*size, rows*size)


    def drawCells(self, numCells, pos_x, pos_y, size):
        # Used to initialise the graphics scene with cells
        transp = QtGui.QColor(0, 0, 0, 0)
        transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        pen = QtGui.QPen(transp)
        brush = QtGui.QBrush(red)  # Color has an opacity
        # Find the smallest number which when squared is larger than numCells
        numSquares = int(math.ceil(numCells ** 0.5))
        # Set the small sqaures to a size smaller than the large ones
        squareSize = size/(1.5*numSquares)
        # Count the cells that are drawn so we can identify them
        count = 0
        for i in range(numSquares):
            for j in range(numSquares):
                brush = QtGui.QBrush(transpRed)  # Make the non existent cells faint
                if count < numCells:
                    # Separate the small squares
                    x = pos_x*size + 0.5*squareSize+1.5*i*squareSize
                    y = pos_y*size + 0.5*squareSize+1.5*j*squareSize
                    cellItem = HTMCell(pos_x, pos_y, count, x, y, squareSize, pen, brush)
                    #cellItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
                    self.cellItems.append(cellItem)  # Add the cells to a list so we can iterate through them to update
                    self.scene.addItem(cellItem)    # Add the cells to the scene
                # Increase the count to keep track of how many cells have been created
                count += 1

    def drawSingleColumn(self, pos_x, pos_y):
        # Draw the cells connected to the selected segment
        print "Column Synapse permanence"
        #transp = QtGui.QColor(0, 0, 0, 0)
        #pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        #transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        #red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        #green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        #blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        # Go through each column. If it is in the synapse list draw it otherwise don't
        for i in range(len(self.columnItems)):
            column_pos_x = self.columnItems[i].pos_x
            column_pos_y = self.columnItems[i].pos_y
            brush = QtGui.QBrush(transpBlue)   # Have to create a brush with a color
            # Check each synapse and draw the connected columns
            currentLayer = self.htm.regionArray[self.level].layerArray[self.layer]
            selectedColumn = currentLayer.columns[pos_y][pos_x]
            for syn in currentLayer.getConnectedSynapses(selectedColumn):
                if syn.pos_x == column_pos_x and syn.pos_y == column_pos_y:
                    #print "     syn x, y= %s,%s Permanence = %s" % (column_pos_x, column_pos_y, syn.permanence)
                    brush.setColor(darkGreen)
            self.columnItems[i].setBrush(brush)
            #self.columnItems[i].setPen(pen)

    def drawSingleCell(self, pos_x, pos_y, cellInd, segmentInd):
        # Draw the cells connected to the selected segment
        print"pos_x,pos_y,cell,seg = %s,%s,%s,%s" % (pos_x, pos_y, cellInd, segmentInd)
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        #transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        #red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        white = QtGui.QColor(0xFF, 0xFF, 0xFF)
        #green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        #darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)

        # Go through each cell. If it is in the synapse list draw it otherwise don't
        print "Synapses start in col x,y = %s, %s" % (pos_x, pos_y)
        for i in range(len(self.cellItems)):
            cell_pos_x = self.cellItems[i].pos_x
            cell_pos_y = self.cellItems[i].pos_y
            cell_cell = self.cellItems[i].cell

            currentLayer = self.htm.regionArray[self.level].layerArray[self.layer]
            selectedColumn = currentLayer.columns[pos_y][pos_x]
            # A brush must be created with a color
            brush = QtGui.QBrush(transpBlue)
            # Check each synapse and draw the connected cells.
            # from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
            synList = currentLayer.getConnectedCellsSegSyns(selectedColumn, cellInd, segmentInd)

            for syn in synList:
                # Save the synapses end cells active times so they can be displayed.
                # synEndColumn = self.htm.regionArray[self.level].layerArray[self.layer].columns[syn.pos_y][syn.pos_x]
                if syn.pos_x == cell_pos_x and syn.pos_y == cell_pos_y and syn.cell == cell_cell:
                    print "     syn x,y,cell= %s,%s,%s Permanence = %s" % (syn.pos_x, syn.pos_y, syn.cell, syn.permanence)
                    brush.setColor(blue)
                # Set this cells color to white to indicate it was selected.
                if pos_x == cell_pos_x and pos_y == cell_pos_y and cellInd == cell_cell:
                    brush.setColor(white)
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)

    def drawColumnInhib(self, pos_x, pos_y):
        # Draw the columns that are inhibited by the selected column at position x,y
        column = self.htm.regionArray[self.level].layerArray[self.layer].columns[pos_y][pos_x]
        htmlayer = self.htm.regionArray[self.level].layerArray[self.layer]
        red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)

        neighbourColumns = htmlayer.neighbours(column)
        for colItem in self.columnItems:
            colItem_pos_x = colItem.pos_x
            colItem_pos_y = colItem.pos_y
            for col in neighbourColumns:

                if col.pos_x == colItem_pos_x and col.pos_y == colItem_pos_y:
                    value = htmlayer.columnActiveStateNow(col.pos_x, col.pos_y)
                    color = QtGui.QColor(0xFF, 0, 0, 0xFF)
                    brush = QtGui.QBrush(QtCore.Qt.red)
                    if (value is False):
                        color = transpBlue
                    elif (value is True):
                        color = darkGreen
                    brush.setColor(color)
                    colItem.setBrush(brush)

    def updateGrid(self):
        layer = self.htm.regionArray[self.level].layerArray[self.layer]
        for i in range(len(self.columnItems)):
            brush = QtGui.QBrush(QtCore.Qt.green)
            brush.setStyle(QtCore.Qt.SolidPattern)
            pos_x = self.columnItems[i].pos_x
            pos_y = self.columnItems[i].pos_y
            # Check if the column at pos_x, pos_y is active now
            value = layer.columnActiveStateNow(pos_x, pos_y)
            if value is False:
                    brush.setColor(QtCore.Qt.red)
                    self.columnItems[i].setBrush(brush)
            if value is True:
                    brush.setColor(QtCore.Qt.green)
                    self.columnItems[i].setBrush(brush)
        self.updateCells()
        # Update the info
        self.updateInfo()

    def updateCells(self):
        currentLayer = self.htm.regionArray[self.level].layerArray[self.layer]
        # Redraw the cells.
        timeStep = currentLayer.timeStep
        transp = QtGui.QColor(0, 0, 0, 0)
        pen = QtGui.QPen(transp, 0, QtCore.Qt.SolidLine)
        #transpRed = QtGui.QColor(0xFF, 0, 0, 0x20)
        #red = QtGui.QColor(0xFF, 0, 0, 0xFF)
        black = QtGui.QColor(0, 0, 0, 0xFF)
        transpBlue = QtGui.QColor(0, 0, 0xFF, 0x30)
        #green = QtGui.QColor(0, 0xFF, 0, 0xFF)
        darkGreen = QtGui.QColor(0, 0x80, 0x40, 0xFF)
        blue = QtGui.QColor(0x40, 0x30, 0xFF, 0xFF)
        for i in range(len(self.cellItems)):
            pos_x = self.cellItems[i].pos_x
            pos_y = self.cellItems[i].pos_y
            cell = self.cellItems[i].cell
            brush = QtGui.QBrush(transp)  # Make the non existent cells faint
            if self.showActiveCells is True:
                if currentLayer.checkCellActive(pos_x, pos_y, cell, timeStep) is True:
                    brush.setColor(blue)
                else:
                    brush.setColor(transpBlue)
            if self.showPredictCells is True:
                if currentLayer.checkCellPredict(pos_x, pos_y, cell, timeStep) is True:
                    brush.setColor(black)
                else:
                    brush.setColor(transpBlue)
            if self.showLearnCells is True:
                if currentLayer.checkCellLearn(pos_x, pos_y, cell, timeStep) is True:
                    brush.setColor(darkGreen)
                else:
                    brush.setColor(transpBlue)
            self.cellItems[i].setBrush(brush)
            self.cellItems[i].setPen(pen)

    def scaleScene(self, scaleSize):
        self.scaleSize = self.scaleSize*scaleSize
        #print "ScaleSize = %s" % self.scaleSize
        self.scale(scaleSize, scaleSize)

    def resizeView(self):
        # Resize the view. This function is called by some qt signals.
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def mousePressEvent(self, event):

        if event.buttons() == QtCore.Qt.LeftButton:
            layer = self.htm.regionArray[self.level].layerArray[self.layer]
            item = self.itemAt(event.x(), event.y())
            self._mousePressed = True

            # If the info item was not clicked then let the user scroll around with the mouse
            if item.__class__.__name__ != "HTMInfo":
                self.dragView = True
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                self._dragPos = event.pos()
                event.accept()

            # The info item can be dragged to be repositioned in the screen
            if item.__class__.__name__ == "HTMInfo":
                self.dragInfo = True
                self._dragPos = event.pos()
                event.accept()

            if item.__class__.__name__ == "HTMCell":
                print "cell"
                print "pos_x,pos_y,cell = %s,%s,%s" % (item.pos_x, item.pos_y, item.cell)
                # Print out the cells current score.
                currentLayer = self.htm.regionArray[self.level].layerArray[self.layer]
                cellsScore = currentLayer.getCellsScore(item.pos_x, item.pos_y, item.cell)
                print "     Cells Score = %s" % cellsScore
                currentLayer.getCellsUpdateSynStruct(item.pos_x, item.pos_y, item.cell)
                numSegments = layer.getNumSegments(item.pos_x, item.pos_y, item.cell)
                self.selectedItem = item
                item_pos = item.pos()
                popup_pos_x = item_pos.x()+self.x()
                popup_pos_y = item_pos.y()+self.y()
                # Create the popup window at a certain position
                self.segmentSelect = popup(event.x(), event.y(), item.cell, numSegments)
                self.segmentSelect.setGeometry(QtCore.QRect(popup_pos_x, popup_pos_y, 200, 200))
                # Create and connect a Slot to the signal from the check box
                self.segmentSelect.segmentSelectedSignal.connect(self.selectedSegmentIndex)
                self.segmentSelect.show()
            if item.__class__.__name__ == "HTMColumn":
                print"column"
                print "pos_x, pos_y = %s, %s" % (item.pos_x, item.pos_y)
                self.updateGrid()
                self.drawColumnInhib(item.pos_x, item.pos_y)
                # Create and emit a signal to tell the HTMInput viewer to draw the
                # input grid squares which are connected by the selected columns connected synapses.
                self.selectedColumn.emit(item.pos_x, item.pos_y)

        if event.buttons() == QtCore.Qt.RightButton:
            # Toggle the view from predicted, learn and active cells.
            if self.showActiveCells is True:
                self.showActiveCells = False
                self.showPredictCells = True
                self.showLearnCells = False
            elif self.showPredictCells is True:
                self.showActiveCells = False
                self.showPredictCells = False
                self.showLearnCells = True
            elif self.showLearnCells is True:
                self.showActiveCells = False
                self.showPredictCells = False
                self.showLearnCells = False
            else:
                self.showActiveCells = True
            self.updateGrid()

    def mouseMoveEvent(self, event):
        if self.dragView is True:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        elif self.dragInfo is True:
            # Enable Dragging of the info widget around the screen
            newPos = event.pos()
            # When the view is zoomed in you need to scale how much the info item
            # moves so it keeps pace with the mouse pointer
            diff = (1/float(self.scaleSize)) * (newPos - self._dragPos)
            self._dragPos = newPos
            self.infoItem.setPos(self.infoItem.pos() + diff)
        else:
            super(HTMGridViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self.setCursor(QtCore.Qt.OpenHandCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
            self.dragView = False
            self._mousePressed = False
            self.dragInfo = False
        #if event.button() == QtCore.Qt.RightButton:
        #        super(HTMGridViewer, self).mouseReleaseEvent(event)
        super(HTMGridViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Fit the scene into the view
        self.resizeView()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control and not self.dragView:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        # Toggle through the differnt regions (layers) in the HTM
        elif event.key() == QtCore.Qt.Key_R:
            self.layer = self.layer + 1
            if self.layer >= self.htm.regionArray[self.level].numLayers:
                self.layer = 0
            #print "Layer %s" % layer
            #Update the HTM veiwer
            self.updateGrid()
        # Toggle through the differnt levels in the HTM
        elif event.key() == QtCore.Qt.Key_L:
            self.level = self.level + 1
            if self.level >= self.htm.numLevels:
                self.level = 0
            #print "Level %s" % level
            #Update the HTM veiwer
            self.updateGrid()
        # The number keys 1,2,3 select the show active, show pred and show learn cells.
        elif event.key() == QtCore.Qt.Key_1:
            self.showActiveCells = True
            self.showPredictCells = False
            self.showLearnCells = False
            self.updateGrid()
        elif event.key() == QtCore.Qt.Key_2:
            self.showActiveCells = False
            self.showPredictCells = True
            self.showLearnCells = False
            self.updateGrid()
        elif event.key() == QtCore.Qt.Key_3:
            self.showActiveCells = False
            self.showPredictCells = False
            self.showLearnCells = True
            self.updateGrid()
        else:
            super(HTMGridViewer, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            if not self.dragView:
                self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            super(HTMGridViewer, self).keyPressEvent(event)

    def wheelEvent(self, event):
        factor = 1.2
        if event.delta() < 0:
            factor = 1.0 / factor
        self.scaleScene(factor)


class HTMNetwork(QtGui.QWidget):
    # Creates a HTM network.

    def __init__(self, HTMObject, InputCreator):
        super(HTMNetwork, self).__init__()
        # Pass the HTM network to view if it was given.
        self.initUI(HTMObject, InputCreator)

    def initUI(self, HTMObject, InputCreator):

        # Use the specified HTM to set local variables
        self.htm = HTMObject
        self.iteration = self.htm.regionArray[0].layerArray[0].timeStep
        self.origIteration = self.iteration  # Stores the iteration for the previous saved HTM
        self.numLevels = self.htm.numLevels  # The number of levels.
        self.numCells = self.htm.cellsPerColumn  # The number of cells in a column.
        self.width = self.htm.width  # The width of the columns in the HTM 2D array
        self.height = self.htm.height  # The height of the columns in the HTM 2D array
        self.inputWidth = len(self.htm.regionArray[0].layerArray[0].Input[0])*self.numCells
        self.inputHeight = len(self.htm.regionArray[0].layerArray[0].Input)
        self.InputCreator = InputCreator

        # Create the widgets to veiw htm layers and their inputs.
        # Each one views a different level of the htm
        self.HTMNetworkGrid = []
        self.inputGrid = []
        countNum = 0
        for i in range(self.numLevels):
            # Get the number of layers in the level in the htm.
            numLayers = self.htm.regionArray[i].numLayers
            for j in range(numLayers):
                # HTM grid veiwer widgets.
                self.HTMNetworkGrid.append(HTMGridViewer(self.htm))
                self.inputGrid.append(HTMInput(self.htm, self.HTMNetworkGrid[countNum]))
                # Setup each viewer to display the correct level and layer
                self.setLevel(self.HTMNetworkGrid[-1], i)
                self.setLevel(self.inputGrid[-1], i)
                self.setLayer(self.HTMNetworkGrid[-1], j)
                self.setLayer(self.inputGrid[-1], j)
                countNum += 1

        # Used to create and save new views
        self.markedHTMViews = []

        self.scaleFactor = 0.2    # How much to scale the grids by
        self.grid = None    # This is the layout holding the frames.
        self.frameSplitter = None  # This allows widgets to be resized by dragging the mouse
        self.frameSplitterH = None
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None

        self.btn1 = None  # HTM +
        self.btn2 = None  # HTM -
        self.btn3 = None  # All HTM
        self.btn4 = None  # Nsteps
        self.btn5 = None  # Step
        self.btn6 = None  # In +
        self.btn7 = None  # In -
        self.btn8 = None  # Active
        self.btn9 = None  # Predict
        self.btn10 = None  # Learn
        self.btn11 = None  # Save
        self.btn12 = None  # Load
        self.btn13 = None  # Level select
        self.btn14 = None  # Mark state
        self.makeFrames()
        self.makeButtons()
        #self.setWindowTitle('Main window')
        self.show()

    def setInput(self, width, height):
        input = np.array([[0 for i in range(width)] for j in range(height)])
        return input

    def makeButtons(self):
        #self.btn1 = QtGui.QPushButton("HTM +", self)
        #self.btn1.clicked.connect(self.HTMzoomIn)
        #self.btn2 = QtGui.QPushButton("HTM -", self)
        #self.btn2.clicked.connect(self.HTMzoomOut)
        self.btn3 = QtGui.QPushButton("All HTM", self)
        self.btn3.clicked.connect(self.showAllHTM)
        self.btn4 = QtGui.QPushButton("n steps", self)
        self.btn4.clicked.connect(self.nSteps)
        self.btn5 = QtGui.QPushButton("step", self)
        self.btn5.clicked.connect(self.oneStep)
        #self.btn6 = QtGui.QPushButton("In +", self)
        #self.btn6.clicked.connect(self.inputZoomIn)
        #self.btn7 = QtGui.QPushButton("IN -", self)
        #self.btn7.clicked.connect(self.inputZoomOut)
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
        # Create the layer dropDown
        self.layerDropDown()

        # Add the dropdown menu to the screens top frame
        # addWidget(QWidget, row, column, rowSpan, columnSpan)
        self.grid.addWidget(self.btn5, 1, 4, 1, 1)
        self.grid.addWidget(self.btn4, 1, 5, 1, 1)
        self.grid.addWidget(self.btn11, 2, 1, 1, 1)
        self.grid.addWidget(self.btn12, 2, 2, 1, 1)
        self.grid.addWidget(self.btn8, 1, 6, 1, 1)
        self.grid.addWidget(self.btn9, 1, 7, 1, 1)
        self.grid.addWidget(self.btn10, 1, 8, 1, 1)
        self.grid.addWidget(self.btn3, 2, 5, 1, 1)
        #self.grid.addWidget(self.btn6, 3, 1, 1, 1)
        #self.grid.addWidget(self.btn7, 3, 2, 1, 1)
        #self.grid.addWidget(self.btn1, 3, 5, 1, 1)
        #self.grid.addWidget(self.btn2, 3, 6, 1, 1)
        self.grid.addWidget(self.btn14, 2, 6, 1, 1)

    def levelDropDown(self):
        # Create a drop down button to select the level in the HTM to draw.
        self.btn13 = QtGui.QToolButton(self)
        self.btn13.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        self.btn13.setMenu(QtGui.QMenu(self.btn13))
        self.setLevelAction = QtGui.QWidgetAction(self.btn13)
        self.levelList = levelPopup(self.htm.numLevels)
        self.setLevelAction.setDefaultWidget(self.levelList)
        self.btn13.menu().addAction(self.setLevelAction)
        # Create and connect a Slot to the signal from the check box
        #self.levelList.levelSelectedSignal.connect(self.setLevel)
        # Add the dropdown menu to the screens top frame
        self.grid.addWidget(self.btn13, 1, 1, 1, 1)

    def layerDropDown(self):
        # Create a drop down button to select the layer in the HTM level to draw.
        self.btn15 = QtGui.QToolButton(self)
        self.btn15.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        self.btn15.setMenu(QtGui.QMenu(self.btn15))
        self.setLayerAction = QtGui.QWidgetAction(self.btn15)
        # Use the number of layers from the first region. This will
        # only work if all regions have the same number of layers.
        self.layerList = layerPopup(self.htm.regionArray[0].numLayers)
        self.setLayerAction.setDefaultWidget(self.layerList)
        self.btn15.menu().addAction(self.setLayerAction)
        # Create and connect a Slot to the signal from the check box
        #self.layerList.levelSelectedSignal.connect(self.setLayer)
        # Add the dropdown menu to the screens top frame
        self.grid.addWidget(self.btn15, 1, 2, 1, 2)

    def showAllHTM(self):
        # Draw the entire HTM netwrok. This is used if previously just a
        # single cells segment connection was being shown
        for i in range(len(self.HTMNetworkGrid)):
            self.HTMNetworkGrid[i].showAllHTM = True
            self.HTMNetworkGrid[i].updateGrid()

    def markHTM(self):
        # Mark the current state of the HTM by creatng an new view to view the current state.
        # Copy the current htm network once. The newly created views will use this one copy of the htm to
        # display different layers and levels.
        newCopiedHTM = copy.deepcopy(self.htm)
        for HTMView in self.HTMNetworkGrid:
            self.markedHTMViews.append(HTMGridViewer(self.htm))
            # Get the newest view added to the end of the array
            newHTMView = self.markedHTMViews[-1]
            # Use the HTMGridVeiw objects that has been appended to the end of the list
            newHTMView.htm = newCopiedHTM
            # Update the view settings
            newHTMView.showActiveCells = HTMView.showActiveCells
            newHTMView.showLearnCells = HTMView.showLearnCells
            newHTMView.showPredictCells = HTMView.showPredictCells
            # Set the current level and region to draw
            self.setLayer(newHTMView, HTMView.layer)
            self.setLevel(newHTMView, HTMView.level)

            # Redraw the new view
            newHTMView.updateGrid()

    def showActiveCells(self):
        # Toggle between showing the active cells or not
        for i in range(len(self.HTMNetworkGrid)):
            if self.HTMNetworkGrid[i].showActiveCells is True:
                self.HTMNetworkGrid[i].showActiveCells = False
            else:
                self.HTMNetworkGrid[i].showActiveCells = True
                # Don't show the learning or predictive cells
                self.HTMNetworkGrid[i].showPredictCells = False
                self.HTMNetworkGrid[i].showLearnCells = False
            # Update the HTMnetworkgrid veiwer
            self.HTMNetworkGrid[i].updateGrid()

    def showPredictCells(self):
        # Toggle between showing the predicting cells or not
        for i in range(len(self.HTMNetworkGrid)):
            if self.HTMNetworkGrid[i].showPredictCells is True:
                self.HTMNetworkGrid[i].showPredictCells = False
            else:
                self.HTMNetworkGrid[i].showPredictCells = True
                # Don't show the learning or active cells
                self.HTMNetworkGrid[i].showActiveCells = False
                self.HTMNetworkGrid[i].showLearnCells = False
            # Update the HTMnetworkgrid veiwer
            self.HTMNetworkGrid[i].updateGrid()

    def showLearnCells(self):
        # Toggle between showing the learning cells or not
        for i in range(len(self.HTMNetworkGrid)):
            if self.HTMNetworkGrid[i].showLearnCells is True:
                self.HTMNetworkGrid[i].showLearnCells = False
            else:
                self.HTMNetworkGrid[i].showLearnCells = True
                # Don't show the learning or active cells
                self.HTMNetworkGrid[i].showActiveCells = False
                self.HTMNetworkGrid[i].showPredictCells = False
            # Update the HTMnetworkgrid veiwer
            self.HTMNetworkGrid[i].updateGrid()

    def keyPressEvent(self, event):
        # Spacebar is perform next step
        if event.key() == QtCore.Qt.Key_Space:
            self.step(True)

    def HTMzoomIn(self):
        for i in range(self.numLevels):
            self.HTMNetworkGrid.scaleScene(1+self.scaleFactor)

    def HTMzoomOut(self):
        for i in range(self.numLevels):
            self.HTMNetworkGrid.scaleScene(1-self.scaleFactor)

    def inputZoomIn(self):
        for i in range(self.numLevels):
            self.inputGrid[i].scaleScene(1+self.scaleFactor)

    def inputZoomOut(self):
        for i in range(self.numLevels):
            self.inputGrid[i].scaleScene(1-self.scaleFactor)

    def makeFrames(self):
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(1)
        # Create multiple frame splitters to store each of the HTMviews
        # Their is one horizontal splitter for each level
        self.frameSplitter = QtGui.QSplitter(self)
        self.frameSplitter.setOrientation(QtCore.Qt.Vertical)
        self.frameSplitterH = [QtGui.QSplitter(self) for i in range(self.numLevels)]

        # addWidget(QWidget, row, column, rowSpan, columnSpan)
        # Set the minimum size for the HTM veiwer (row , minsize in pixels)
        self.grid.setRowMinimumHeight(6, self.height)
        self.grid.addWidget(self.frameSplitter, 3, 1, 4, 8)

        # Add each pair of input and HTM layer views to the splitter frames
        # There is an input and htm view for each layer in every level.
        countNum = 0
        for i in range(self.numLevels):
            self.frameSplitterH[i].setOrientation(QtCore.Qt.Horizontal)
            # Get the number of layers in the level in the htm.
            numLayers = self.htm.regionArray[i].numLayers
            for j in range(numLayers):
                # Add to each horizontal splitter every layer in a particular level
                self.frameSplitterH[i].addWidget(self.inputGrid[countNum])
                self.frameSplitterH[i].addWidget(self.HTMNetworkGrid[countNum])
                countNum += 1
            #Add each horizontal frame splitter to the vertical one
            self.frameSplitter.addWidget(self.frameSplitterH[i])

        self.setLayout(self.grid)

    def setLevel(self, view, level):
        # Set the given view to display the given level.
        # The view can be for the htm or the input.
        # Set the level for the HTMVeiwer to draw.
        print "Level set to %s" % level

        # Update the columns and cells of the viewer
        view.level = level
        view.drawGrid(view.size)
        view.updateGrid()

    def setLayer(self, view, layer):
        # Set the given view to display the given layer.
        # The view can be for the htm or the input.
        # Set the layer for the HTMVeiwer to draw.
        print "Layer set to %s" % layer

        # Update the columns and cells of the viewer
        view.layer = layer
        view.drawGrid(view.size)
        view.updateGrid()

    def saveHTM(self):
        self.htm.saveRegions()
        self.origIteration = self.iteration
        print "Saved HTM layers"

    def loadHTM(self):
        # We need to make sure the GUI points to the correct object
        origHTM = self.htm.loadRegions()
        self.htm.regionArray = origHTM
        self.iteration = self.origIteration
        for i in range(len(self.HTMNetworkGrid)):
            self.HTMNetworkGrid[i].iteration = self.origIteration
            # Update the veiws to display the newly loaded htm network.
            self.HTMNetworkGrid[i].updateGrid()
            self.inputGrid[i].updateGrid()
        print "loaded HTM layers"

    def oneStep(self):
        # Used as a call back for the one step button.
        # This is done so the True argument is passed and the viewers are then updated.
        self.step(True)

    def nSteps(self):
        numSteps, ok = QtGui.QInputDialog.getInt(self, 'number of steps', 'steps:')
        if ok:
            print numSteps
            # Minus one from the number of steps since we only update
            # the veiwer on the last step.
            for i in range(numSteps-1):
                self.step(False)
            # Update the viewer on the last step.
            self.step(True)

    def step(self, updateViewer):
        # Update the inputs and run them through the HTM levels just once.

        print "NEW TimeStep"
        #######################################################################
        # PART 1 MAKE NEW INPUT FOR LEVEL 0
        ########################################################################
        print "PART 1 Update Input"
        # Get the command output in the form of an SDR.
        # The command output will be from the top layer in the bottom level.
        topLayer = self.htm.regionArray[0].numLayers-1
        commandGrid = self.htm.regionArray[0].commandSpaceOutput(topLayer)

        # Use the command output from the top layer to give to the simulator so it can
        # work out a new state.
        self.InputCreator.step(commandGrid)

        # Update the thalamus in each command layer within each level.
        # This selects a new command to try.
        self.htm.updateAllThalamus()

        #######################################################################
        # PART 2 RUN THE NEW INPUT THROUGHT THE HTM
        #######################################################################
        print "PART 2 Update HTM"
        self.iteration += 1  # Increase the time
        # Update the HTM input and run through the
        newInput = self.InputCreator.createSimGrid()
        self.htm.spatialTemporal(newInput)

        #######################################################################
        # PART 3 REWARD THE THALAMUS
        #######################################################################
        # Update the Thalamus class by either rewarding it or not
        # Get a reward from the simulation.
        # Update Each Thalamus in the HTM heirarchy.
        currentReward = self.InputCreator.getReward()
        self.htm.rewardAllThalamus(currentReward)

        #######################################################################
        # Check if the view should be updated
        if updateViewer is True:
            countNum = 0
            for i in range(self.numLevels):
                # Get the number of layers in the level in the htm.
                numLayers = self.htm.regionArray[i].numLayers
                for j in range(numLayers):
                    # Set the input viewers array to self.input
                    self.inputGrid[countNum].updateGrid()
                    # Update the columns and cells of the HTM viewers
                    self.HTMNetworkGrid[countNum].updateGrid()
                    countNum += 1

        print "Number of TimeSteps=%s" % (self.iteration)
        print "------------------------------------------"


class HTMGui(QtGui.QMainWindow):
    # Create a signal to to signal that the window size has changed.
    windowSizeChanged = QtCore.pyqtSignal()

    def __init__(self, HTMInput, InputCreator):
        super(HTMGui, self).__init__()

        self.initUI(HTMInput, InputCreator)

    def initUI(self, HTMInput, InputCreator):
        # Connect up a close event to a function
        self.closeEvent = self.closeEvent
        layout = QtGui.QHBoxLayout()
        # Create the HTM Viewing widget and pass in the HTM object to view
        self.HTMWidget = HTMNetwork(HTMInput, InputCreator)
        layout.addWidget(self.HTMWidget)
        self.statusBar().showMessage('Ready')

        newInputAction = QtGui.QAction(QtGui.QIcon('grid.png'),
                                       '&New Input', self)
        newInputAction.setStatusTip('create a new input')
        #newInputAction.triggered.connect(HTMWidget.createNewInput())
        drawLevelAction = QtGui.QAction(QtGui.QIcon('grid.png'),
                                        '&Level', self)
        drawLevelAction.setStatusTip('draw Level x')
        #drawLevelAction.triggered.connect(HTMWidget.drawLevel())

        #self.toolbar = self.addToolBar('create a new input')
        #self.toolbar.addAction(newInputAction)
        #self.toolbar = self.addToolBar('draw a different Level')
        #self.toolbar.addAction(drawLevelAction)

        # Connect this signal to every htm input and htm network widget.
        self.connectWindowResizeEvent()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        ViewMenu = menubar.addMenu('&View')
        fileMenu.addAction(newInputAction)
        ViewMenu.addAction(drawLevelAction)

        self.setGeometry(600, 100, 800, 900)
        self.setWindowTitle('HTM')
        self.setCentralWidget(self.HTMWidget)
        self.show()

    def connectWindowResizeEvent(self):
        # Need to iterate through all the input and htm widgets and
        # connect the resize signal to a slot to tell them to resize their view.
        numInputWidgets = len(self.HTMWidget.inputGrid)
        numHTMWidgets = len(self.HTMWidget.HTMNetworkGrid)
        for i in range(numInputWidgets):
            self.windowSizeChanged.connect(self.HTMWidget.inputGrid[i].resizeView)
        for j in range(numHTMWidgets):
            self.windowSizeChanged.connect(self.HTMWidget.HTMNetworkGrid[j].resizeView)

    def resizeEvent(self, resizeEvent):
        # The screen has been resized.
        # Send a signal to resize all the input and htm views so they
        # fit the new screen size.
        self.windowSizeChanged.emit()

    def closeEvent(self, event):
        print "Closing"
        self.deleteLater()
        self.close()
        #self.destroyed()

