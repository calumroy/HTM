import sys
import GUI_HTM
from PyQt4 import QtGui


def startHtmGui(htm, InputCreator):
    # A function to load the HTM GUI.
    app = QtGui.QApplication.instance()  # checks if QApplication already exists
    if not app:  # create QApplication if it doesnt exist
        app = QtGui.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    htmGui = GUI_HTM.HTMGui(htm, InputCreator)
    app.exec_()
