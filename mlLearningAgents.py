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
import math

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


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

        "*** YOUR CODE HERE ***"
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood()
        self.score = state.getScore()

    def __hash__(self):
        return hash((self.pacmanPosition, tuple(self.ghostPositions), self.food))

    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return self.pacmanPosition == other.pacmanPosition and \
               self.ghostPositions == other.ghostPositions and \
               self.food == other.food


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
        self.qTable = util.Counter()
        self.countsTable = util.Counter()
        self.wonGames = 0
        self.initialNumPellets = 0
        self.onePelletCollections = 0

    def registerInitialState(self, state: GameState):
        self.initialNumPellets = state.getNumFood()

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

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        return self.qTable[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        legalActions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        q_values = [self.getQValue(state, action) for action in legalActions]
        return max(q_values)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        q_value = self.getQValue(state, action)
        max_q_value_next = self.maxQValue(nextState)
        self.qTable[(state, action)] = q_value + self.alpha * (reward + self.gamma * max_q_value_next - q_value)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        self.countsTable[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        return self.countsTable[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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

        # Variant of the epsilon-greedy exploration strategy, exploration bonus is inversely proportional to number of times a given action has been taken in the given state,
        # WIN RATE: 0.71
        return utility + self.epsilon / (1 + counts)

        # Exploration fn based on UCB, exploration bonus decays with square root of ratio between the log of episodesSoFar and the number
        # of times the action has been taken in the given state i.e. counts
        # WIN RATE: 0.68
        # return utility + self.epsilon * math.sqrt(2 * math.log(self.getEpisodesSoFar() + 1) / (counts + 1))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help understand the inputs
        # print("Legal moves: ", legal)
        # print("Pacman position: ", state.getPacmanPosition())
        # print("Ghost positions:", state.getGhostPositions())
        # print("Food locations: ")
        # print(state.getFood())
        # print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        if util.flipCoin(self.epsilon):
            return random.choice(legal)
        else:
            q_values = [(self.getQValue(stateFeatures, action) +
                         self.explorationFn(self.getQValue(stateFeatures, action),
                                            self.getCount(stateFeatures, action)),
                         action) for action in legal]
            _, action = max(q_values)

            # Update the Q-table and counts after taking the action
        nextState = state.generatePacmanSuccessor(action)
        nextStateFeatures = GameStateFeatures(nextState)
        reward = self.computeReward(state, nextState)
        self.learn(stateFeatures, action, reward, nextStateFeatures)
        self.updateCount(stateFeatures, action)
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        episode = self.getEpisodesSoFar()
        score = state.getScore()
        pellets_eaten = self.initialNumPellets - state.getNumFood()
        # Track number of games won and the number of one pellet collections
        if state.isWin():
            self.wonGames += 1
        if pellets_eaten == 1:
            self.onePelletCollections += 1
        print(f"Episode {episode}: Score = {score}, pellets eaten: {pellets_eaten}")

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print(f"Number of won games: {self.wonGames}")
            print(f"Number of one pellet collections {self.onePelletCollections}")
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
