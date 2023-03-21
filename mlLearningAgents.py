# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
import numpy as np


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.features = []
        self.width = state.getFood().width
        self.height = state.getFood().height
        
        # Extract features from the state
        self.features.extend(self.oneHotFoodVector(state))
        self.features.extend(self.oneHotGhostVector(state.getGhostPositions()))
        self.features.extend(self.oneHotPacmanVector(state.getPacmanPosition()))
        
    def oneHotFoodVector(self, state: GameState):
        food = []
        for i in state.getFood():
            for j in i:
                if j:
                    food.append(1)
                else:
                    food.append(0)
        return food

    def oneHotPacmanVector(self, position):
        feature = [0] * self.width * self.height
        index = position[0] * self.width + position[1]
        feature[int(index)] = 1
        return feature
    
    def oneHotGhostVector(self, positions):
        feature = [0] * self.width * self.height
        for position in positions:
            index = position[0] * self.width + position[1]
            feature[int(index)] = 1
        return feature
    

class DQN:
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.01, discount_factor: float = 0.99):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, 128) / np.sqrt(input_size)
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, output_size) / np.sqrt(128)
        self.b2 = np.zeros((1, output_size))
        self.a1 = None
    
    def forward(self, state):
        """
        Performs a forward pass through the network and returns the Q-values
        for each action given the current state

        """
        #print(state)
        z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.tanh(z1)
        z2 = np.dot(self.a1, self.W2) + self.b2
        return z2[0]
    
    def backward(self, state, action_idx, reward, next_state):
        q_values = self.forward(state)
        next_q_values = self.forward(next_state)

        # Target Q-value
        target = reward + self.discount_factor * np.max(next_q_values)

        # Compute the loss
        loss = 0.5 * (target - q_values[action_idx]) ** 2

        # Compute gradients
        delta = np.zeros(self.output_size)
        delta[action_idx] = q_values[action_idx] - target
        dW2 = np.outer(self.a1, delta)
        db2 = delta
        da1 = np.dot(delta, self.W2.T)
        dz1 = (1 - self.a1 ** 2) * da1
        dW1 = np.outer(state, dz1)
        db1 = dz1

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        return loss
    
class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.QNet = DQN(147, 5)

        #self.transitions = {}
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        print("learn")
        loss = self.QNet.backward(state, action, reward, nextState, 0)
        util.raiseNotDefined()

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def actionToOneHot(self, action: Directions) -> np.ndarray:
        oneHot = np.zeros(5)
        if action == Directions.NORTH:
            oneHot[0] = int(1)
        elif action == Directions.SOUTH:
            oneHot[1] = int(1)
        elif action == Directions.EAST:
            oneHot[2] = int(1)
        elif action == Directions.WEST:
            oneHot[3] = int(1)
        elif action == Directions.STOP:
            oneHot[4] = int(1)
        return oneHot
    
    def legalToOneHot(self, actions: Directions) -> np.ndarray:
        oneHot = np.zeros(5)
        if Directions.NORTH in actions:
            oneHot[0] = int(1)
        if Directions.SOUTH in actions:
            oneHot[1] = int(1)   
        if Directions.EAST in actions:
            oneHot[2] = int(1)
        if Directions.WEST in actions:
            oneHot[3] = int(1)
        if Directions.STOP in actions:
            oneHot[4] = int(1)
        return oneHot
    
    def oneHotToAction(self, oneHot: np.ndarray) -> Directions:
        if oneHot[0] == 1:
            return Directions.NORTH
        elif oneHot[1] == 1:
            return Directions.SOUTH
        elif oneHot[2] == 1:
            return Directions.EAST
        elif oneHot[3] == 1:
            return Directions.WEST
        elif oneHot[4] == 1:
            return Directions.STOP

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take

        """
        new_score = state.getScore()
        reward = new_score - self.lastScore
        stateFeatures = GameStateFeatures(state).features

        if self.lastState is not None:
            self.lastAction = self.actionToOneHot(self.lastAction)
            transition = {"last_state": self.lastState, "last_action": self.lastAction, "state": stateFeatures}
            loss = self.QNet.backward(transition["last_state"], np.argmax(transition["last_action"]), reward, transition["state"])
            
        one_hot = np.zeros(5)
        values = self.QNet.forward(stateFeatures)

        legal = state.getLegalPacmanActions()
        legal_one_hot = self.legalToOneHot(legal)
        for i in range(len(legal_one_hot)):
            if legal_one_hot[i] == 0:
                values[i] = -10000000
            
        values[4] = -10000000
        q_max = np.argmax(values)
        one_hot[q_max] = 1

        if np.random.rand() < 0.1:
            action = np.random.choice(legal)
        else:
            action = self.oneHotToAction(one_hot)

        self.lastAction = action
        self.lastState = stateFeatures
        self.lastScore = new_score

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

        last_action = self.actionToOneHot(self.lastAction)

        if state.isLose():
            self.QNet.backward(self.lastState, np.argmax(last_action), -100, GameStateFeatures(state).features)
        else:
            self.QNet.backward(self.lastState, np.argmax(last_action), 100, GameStateFeatures(state).features)
