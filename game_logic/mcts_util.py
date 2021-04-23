from math import log, sqrt
import numpy as np

class MCTSNode:
    def __init__(self, parent, action, player, numberOfAgents):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = np.zeros(numberOfAgents)
        self.player = player



def ucb(node, C=sqrt(2)):
    return node.value[node.player] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)