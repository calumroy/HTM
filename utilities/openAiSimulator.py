
import numpy as np
import math
from encoders import simple_htm_encoder as encoder
import random

class openAiSimulator:
    '''
    A class used to run open ai simulation and pass the inputs and
    outputs to a htm instance. It encodes the observation, reward and
    action variable values into an input binary matrix consisting of
    straight vertical lines. It then feeds these to a HTM network.


    '''
    def __init__(self, open_ai_env, num_episodes, max_time_per_epi,
                 width, height, random_actions):

        # Store a refenrence to the open_ai simualtion that is being run.
        self.open_ai_env = open_ai_env
        # Also store the number of timesteps per episode and the num of episodes to run.
        self.num_episodes = num_episodes
        self.i_episode = 0
        self.timeStep = 0
        self.max_time_per_epi = max_time_per_epi
        # The width and height are the size of the output binary matrix.
        self.width = width
        self.height = height
        # A flag to indicate if random actions should be taken.
        self.random_actions = random_actions
        # How many timesteps should pass before a new random action is taken.
        self.random_action_period = 10
        # Store the previous random action
        self.rand_action = None

        # THe min and max values indicate the minimum and maximum values that
        # the input variables are going to have. They will be an array if more
        # then one input is to be encoded.
        # Get the max and min values of the observation space from the simulation
        max_obs = (self.open_ai_env.observation_space.high)
        min_obs = (self.open_ai_env.observation_space.low)
        print "max_obs = %s" % max_obs
        print "min_obs = %s" % min_obs
        self.min_val = min_obs
        self.max_val = max_obs

        # Add the min and max values of the action to the min/max_vals arrays
        self.maxAcc = 2
        self.minAcc = 0
        self.max_val = np.append(self.max_val, self.maxAcc)
        self.min_val = np.append(self.min_val, self.minAcc)
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()

        self.encoder = encoder.simpleHtmEncoder(self.width, self.height, self.min_val, self.max_val)

        # Initialise the simualtion with this command.
        # Initialise the observation
        self.observation = self.open_ai_env.reset()
        self.reward = 0

        # Initialise the encoded input that is to be sent to the htm.
        # Feed the action and result to the HTM
        action = 1
        encode_vals = np.append(self.observation, action)
        self.newInput = self.encoder.encodeVar(encode_vals)

    def convertSDRtoAcc(self, cellGrid):
        # Convert a sparse distributed representation into an acceleration
        # Each cell output represents a particular acceleration command. In
        # this simple case we are using -1, 0 or 1 m/s^2. Future mapping techniques
        # should just use a random mapping value. The total average of the output
        # accelerations is calculated and returned.

        # The idea is that the HTM will learn about this mapping and eventually choose
        # the right cells so the output from the HTM commands the
        # "correct" acceleration to control the system.

        acceleration = 0
        height = len(cellGrid)
        width = len(cellGrid[0])
        accRange = abs(self.minAcc - self.maxAcc)
        numActiveCells = 0

        for row in range(height):
            for col in range(width):
                if cellGrid[row][col] == 1:
                    # Not a random mapping but close enough. We aren't using completely random
                    # since we want the same mapping each time.
                    #accCell = ((col + row) % accRange) + self.minAcc
                    accCell = float(col)/float(width) * float(accRange) + float(self.minAcc)
                    acceleration += accCell
                    # Calculate the total number of active cells
                    numActiveCells += 1
        # Prevent divide by zero!
        if numActiveCells != 0:
            acceleration = float(acceleration)/float(numActiveCells)
        print "Num of active cells from command = %s" % numActiveCells
        print "Acceleration Command = %s" % acceleration
        return acceleration

    def step(self, cellGrid):
        # Check if a new episode should be run
        if self.timeStep >= self.max_time_per_epi:
            self.i_episode += 1
            self.timeStep = 0
            self.open_ai_env.reset()
            print "NEW Episode = %s" % self.i_episode

        # Render the enviroment
        self.open_ai_env.render()

        # SET THE ACTION
        action = None
        if self.random_actions is False:
            # Convert the input grid input an acceleration first
            action = self.convertSDRtoAcc(cellGrid)
            # Round to the nearest whole number
            action = int(round(action))
        else:
            # Get a new random action every period.
            if self.timeStep % self.random_action_period == 0:
                # Get the min and max action values and choose a random int value between them
                #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
                self.rand_action = int(round(random.uniform(self.min_val[2], self.max_val[2])))
            elif self.rand_action == None:
                self.rand_action = int(round(random.uniform(self.min_val[2], self.max_val[2])))
            action = self.rand_action
        print "ACTION = %s" % action

        # Perform a single simulation time step.
        self.observation, self.reward, done, info = self.open_ai_env.step(action)

        # Negate the reward since this environemnt give negative rewards
        self.reward = -self.reward - 1
        print "REWARD = %s" % self.reward
        # Feed the action and result to the HTM
        encode_vals = np.append(self.observation, action)
        #from PyQt4.QtCore import pyqtRemoveInputHook; import ipdb; pyqtRemoveInputHook(); ipdb.set_trace()
        self.newInput = self.encoder.encodeVar(encode_vals)

        # Check if the episode is done
        if done:
            print("Episode finished after {} timesteps".format(self.timeStep+1))

        # Update the timeSteps and episode number.
        self.timeStep += 1

    def getReward(self):
        # Required function for a InputCreator class
        return self.reward

    def createSimGrid(self):
        return self.newInput
