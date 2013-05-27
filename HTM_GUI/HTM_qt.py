#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode PyQt4 tutorial 

This example draws three rectangles in three
different colors. 

author: Jan Bodnar
website: zetcode.com 
last edited: September 2011
"""
import numpy as np
import random
import sys
from PyQt4 import QtGui, QtCore
import HTM_V1

class HTMGrid(QtGui.QWidget):
    
    def __init__(self):
        super(HTMGrid, self).__init__()
        
        self.initUI()
        
    def initUI(self):      
        #sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)

        self.setGeometry(300, 300, 350, 100)
        self.setWindowTitle('Colors')
	self.setGeometry(QtCore.QRect(0, 0, 400, 400))

	self.array = np.array([[random.randint(0,1) for i in range(4)] for i in range(4)])


        self.show()

    def paintEvent(self, e):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawGrid(qp)
        qp.end()
        
    def drawGrid(self, qp):
        color = QtGui.QColor(0, 0, 0)
        color.setNamedColor('#d4d4d4')
        qp.setPen(color)
	for i in range(4): #0 to 3
		for j in range(4):
			v = self.array[i][j]
			if v == 0:
                                qp.setBrush(QtGui.QColor(200, 0, 0))
			else:
				qp.setBrush(QtGui.QColor(0, 200, 0))        				
                        qp.drawRect(i*50, j*50, 50, 50)

        
class Example(QtGui.QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):

        grid = HTMGrid()
        self.setCentralWidget(grid)

        exitAction = QtGui.QAction(QtGui.QIcon('grid.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)


        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Main window')
        self.show()
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


