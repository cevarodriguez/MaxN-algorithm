import random
from .game import Agent

class RandomPacman(Agent):
  def getAction(self, gameState):    
    legalActions = gameState.getLegalActions(0)    
    return random.choice(legalActions)
    
