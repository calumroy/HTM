

'''
This python script creates and runs the Balancer HTM simulation

'''

import GUI_HTM
from PyQt4 import QtGui
import sys
import HTM_Balancer as HTM_V
import Inverted_Pendulum


def main():
    app = QtGui.QApplication(sys.argv)

    numLevels = 2  # The number of levels.
    numCells = 3  # The number of cells in a column.
    width = 8  # The width of the columns in the HTM 2D array
    height = 26  # The height of the columns in the HTM 2D array

    # This makes the input the same size as the feedback command grids
    # It doesn't have to be though
    inputWidth = width*numCells
    inputHeight = height

    # Create an Input object
    InputCreator = Inverted_Pendulum.InvertedPendulum(inputWidth, inputHeight)

    # Create a HTM object
    htm = HTM_V.HTM(numLevels, InputCreator.createSimGrid(), width, height, numCells)

    ex = GUI_HTM.HTMGui(htm, InputCreator)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
