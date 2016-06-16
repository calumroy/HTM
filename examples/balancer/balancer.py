

'''
This python script creates and runs the Balancer HTM simulation

'''

from PyQt4 import QtGui
import sys
import json
from Simulators import Inverted_Pendulum
from HTM_network import HTM_network
from HTM_GUI import GUI_HTM
#GUI_HTM

def main():
    app = QtGui.QApplication(sys.argv)

    # Open and import the parameters .json file
    with open('balancer.json', 'r') as paramsFile:
        params = json.load(paramsFile)

    # This makes the input the same size as the feedback command grids
    # It doesn't have to be though.
    inputWidth = params['HTM']['columnArrayWidth']*params['HTM']['cellsPerColumn']
    inputHeight = int(params['HTM']['columnArrayHeight']/2)

    # Create an Input object
    InputCreator = Inverted_Pendulum.InvertedPendulum(inputWidth, inputHeight)

    # Create a HTM object
    htm = HTM_network.HTM(InputCreator.createSimGrid(), params)

    # Create and run the GUI
    ex = GUI_HTM.HTMGui(htm, InputCreator)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
