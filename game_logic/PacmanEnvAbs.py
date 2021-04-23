import gym
from gym import error, spaces, utils
from gym.utils import seeding
from functools import reduce
import copy
import numpy as np
import sys
import random
from .keyboardAgents import KeyboardAgent
from .ghostAgents import DirectionalGhost, RandomGhost
from . import textDisplay, graphicsDisplay
from .layout import getLayout
from .gameExtended import ClassicGameRulesExtended, GameExtended
from .game import Game, Directions

_directions = {Directions.NORTH: (0, 1),
               Directions.SOUTH: (0, -1),
               Directions.EAST:  (1, 0),
               Directions.WEST:  (-1, 0),
               Directions.STOP:  (0, 0)}

_directionsAsList = list(_directions.items())

class PacmanEnvAbs():
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_render=False, layout_name="mediumClassic", view_distance = (2, 2), agents = None):
        self.layouts = dict()
        self.layout_name=layout_name
        if agents is None:
            self.pacman = KeyboardAgent()
            self.ghosts = [RandomGhost(i+1) if i % 2 ==
                        0 else DirectionalGhost(i+1) for i in range(20)]
        self.pacman = agents[0]
        self.ghosts = agents[1:]
        frameTime = 0.03

        textDisplay.SLEEP_TIME = frameTime
        self.display_text = textDisplay.PacmanGraphics()
        self.display_graphics = graphicsDisplay.PacmanGraphics(
            1.0, frameTime=frameTime)

        self.beQuiet = True
        self.game = None
        self.view_distance = view_distance
        self.textGraphics = False
        self.reset(enable_render=enable_render, layout_name=layout_name)

    def _init_game(self, layout, pacman, ghosts, display, catchExceptions=False, timeout=30):
        rules = ClassicGameRulesExtended(timeout)
        if self.beQuiet and not self.enable_render:
            # Suppress output and graphics
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = self.beQuiet
        else:
            gameDisplay = display
            rules.quiet = self.beQuiet
            # rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, self.beQuiet, catchExceptions)
        game.init()
        return game

    def reset(self, enable_render=False, layout_name=None):
        self.enable_render = enable_render
        self.layout_name = layout_name if layout_name is not None else self.layout_name
        if (self.game):
            self.game.close()        

        self.layouts[layout_name] = self.layouts.get(
            layout_name, getLayout(layout_name))

        if (self.textGraphics):
            display = self.display_text
        else:
            display = self.display_graphics

        self.game = self._init_game(
            self.layouts[layout_name], self.pacman, self.ghosts, display)        
        return self.game.state

    def step(self, action, agentIndex):        
        self._check_action(action, agentIndex)
        
        obs = self.game.step(action, agentIndex)

        reward = self._get_rewards()

        obs = self.game.state
        done = self.is_end()
        info = {"win": self.game.state.data._win, "internal_pacman_score": self.game.state.getScore()}
        return obs, reward, done, info

    def _get_rewards(self):
        return self.game.state.get_rewards()

    def _get_num_agents(self):
        return len(self.game.agents)

    def _check_action(self, action, agentIndex):
        if action not in map(lambda x: x[0], _directionsAsList):
            raise Exception('Action not in action_space')
        if action not in self.game.state.getLegalActions(agentIndex):
            raise Exception('Action not in legal actions of the Agent')
        return True

    def close(self):
        if (self.game is not None):
            self.game.close()

    def flatten_obs(self, s):
        return tuple(s.flatten())

    def is_end(self):
        return self.game.gameOver or self.game.state.getScore() < -1500

    def get_legal_actions(self, agentIndex):
        return self.game.state.getLegalActions(agentIndex)