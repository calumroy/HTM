
import numpy as np
import math


class simpleHtmEncoder:
    '''
    A class used to encoder variable values into an input binary matrix consisting of
    straight vertical lines. This can then be fed to a HTM network.
    This encoder if just one variable

    '''
    def __init__(self, width, height, min_val, max_val, numInputs=1):
        # The number of inputs that each encoded pattern will represent.
        # E.g there may be more then one variabel that needs to be encoded.
        self.numInputs = numInputs
        # The width and height are the size of the output binary matrix.
        self.width = width
        self.height = height
        # THe min and max values indicat the minimum and maximum values that
        # the input variables are going to have. They will be an array if more
        # then one input is to be encoded.
        self.min_val = min_val
        self.max_val = max_val

        self.var_value = None

        # Each output matrix or "pattern" is a series of 2dArray grids storing binary values 0 or 1.
        # self.output = [np.array([[0 for i in range(self.width)]
        #                         for j in range(self.height)]
        self.en_inputs = []

    def encodeVar(self, var_value):
        # Take the new input value or array of input values and encode them into
        # a binay matrix. Use vertical lines to represent the new values.
        # Note the new values should be between the min and max values.
        if (type(var_value).__name__ != 'ndarray' and
           type(var_value).__name__ != 'list'):
            self.numInputs = 1
            self.var_value = [var_value]
        else:
            # There is more then one value to be encoded.
            # Make sure we know the range of new input values.
            #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
            assert len(var_value) == len(self.max_val)
            assert len(var_value) == len(self.min_val)
            self.var_value = var_value

        print "var_value  %s" % var_value
        self.en_inputs = np.array([[0 for i in range(self.width)]
                                  for j in range(self.height)])

        # For each value set a portion of a vertical line of grid values to true, representing that value.
        num_v = len(self.var_value)
        portion_height = int(math.floor(float(self.height) / float(num_v)))

        for i_ind in range(num_v):
            section_start = portion_height*i_ind
            section_end = portion_height*(i_ind+1)
            for y in range(section_start, section_end):
                range_vals = (self.max_val[i_ind]-self.min_val[i_ind])
                #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                # Remeber the minus one since numpy arrays start at positon zero.
                y_pos = abs(round(self.var_value[i_ind] * (float(self.width-1) / float(range_vals))))
                for x in range(self.width):
                    if x == y_pos:
                        self.en_inputs[y][x] = 1


        #print "encoded variable val= %s = \n%s" % (var_value, self.en_inputs)
        return self.en_inputs

    def step(self, cellGrid):
        # Required function for a InputCreator class
        pass

    def getReward(self):
        # Required function for a InputCreator class
        reward = 0
        return reward

    def createSimGrid(self):
        self.encodeVar(self.var_value)

