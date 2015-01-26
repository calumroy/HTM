from mock import MagicMock
from mock import patch
from HTM_Balancer import HTM, HTMLayer, HTMRegion, Column
import numpy as np



class test_TemporalPooling:
    def setUp(self):
        self.width = 10
        self.height = 10
        self.cellsPerColumn = 3

        # Create an array of input which will be fed to the htm so it
        # can try to temporarily pool them.
        self.numInputs = 10
        self.inputs = np.array([[[0 for i in range(self.width * self.cellsPerColumn)]
                                for j in range(self.height)] for k in range(self.numInputs)])

        self.setInputs(self.inputs)
        self.htmlayer = HTMLayer(self.inputs[0], self.width, self.height, self.cellsPerColumn)

    def setInputs(self, inputs):
        import ipdb; ipdb.set_trace()

    def test_case1(self):
        pass


