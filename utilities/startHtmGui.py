import sys
from HTM_GUI import GUI_HTM
from PyQt4 import QtGui
from PyQt4.QtCore import pyqtRemoveInputHook

class start_htm_gui:
	def __init__(self):
		self.app = QtGui.QApplication.instance()  # checks if QApplication already exists

	def startHtmGui(self, htm, InputCreator):
	    # A function to load the HTM GUI.
	    
	    if not self.app:  # create QApplication if it doesnt exist
	        self.app = QtGui.QApplication(sys.argv)
	    #self.app.aboutToQuit.connect(self.app.deleteLater)
	    htmGui = GUI_HTM.HTMGui(htm, InputCreator)
	    self.app.exec_()
	    #app.exit()
	    #app.closed()
	    #pyqtRemoveInputHook()
	    #self.app.deleteLater()
	    #del app
	    del htmGui
    
