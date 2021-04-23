import random
from game_logic.game import Agent
from game_logic.gameExtended import GameStateExtended
import numpy as np
import entregables.calcular_distancias as calcular_distancias
from game_logic import mcts_util, game_util

class MaxNAgent(Agent):
  def __init__(self, index, max_depth = 2, unroll_type = "MC", max_unroll_depth= 5, number_of_unrolls = 10, view_distance = (2,2),param_tunner=None):
    super().__init__(index)
    self.max_depth = max_depth
    self.unroll_type = unroll_type
    self.max_unroll_depth = max_unroll_depth
    self.number_of_unrolls = number_of_unrolls
    self.view_distance = view_distance
    self.param_tunner = param_tunner

  def evaluationFunction(self, gameState, agentIndex,Is_End=0,Pacman_Wins=0):  
    if self.param_tunner == None:
      param_tunner={'retorno_inicial':0,'pacman_food':-200,'pacman_capsule':-300,'pacman_ghost':400,'pacman_s_ghost':400,'ghost_pacman':-100,'ghost_food':0}
    else:
      param_tunner=self.param_tunner

    Num_Agents = gameState.getNumAgents()
    retorno_vector=[]
    for Agent in range(Num_Agents):
      processed_obs = game_util.process_state(gameState, self.view_distance, Agent)
  
      if Agent == 0: #Soy PacMan
        retorno=param_tunner['retorno_inicial']

        #Comida mas cercana
        retorno = retorno + param_tunner['pacman_food'] * gameState.getNumFood()
        #Mas comidas activas menos retorno tengo

        #Capsula mas cercana
        retorno = retorno + param_tunner['pacman_capsule'] * len(gameState.getCapsules())
        #Mas capsulas activas menos retorno tengo

        #Fantasma mas cercano
        distancia_minima=float("inf") #Fantasma no Asustado
        distancia_minima_s=float("inf") #Fantasma Asustado

        g_positions=gameState.getGhostPositions()
        g_states=gameState.getGhostStates()

        for index in range(len(g_positions)): #Recorro los fantasmas

          if g_states[index].scaredTimer==0: #Fantasmas no asustados
            distancia = self.calcular_distancia(g_positions[index],gameState.getPacmanPosition())
            if  distancia < distancia_minima:
              distancia_minima = distancia #Distancia minima a fantasma asustado
          
          else: #Fantasmas asustados
            distancia = self.calcular_distancia(g_positions[index],gameState.getPacmanPosition())
            if  distancia < distancia_minima_s:
              distancia_minima_s = distancia #Distancia minima a fantasma asustado

        if distancia_minima != float("inf"): #Penalizo según fantasma mas cercana 
          retorno = retorno + param_tunner['pacman_ghost'] * distancia_minima

        if distancia_minima_s != float("inf"): #Penalizo según fantasma asustado mas cercano
          retorno = retorno + param_tunner['pacman_s_ghost'] / distancia_minima_s


        if Is_End:
          if Pacman_Wins:
            retorno = float('inf') #Pacman Gana, es mejor estado
          else:
            retorno = -float('inf')#Pacman Pierde, es peor estado

      else: #Soy un Fantasma
        retorno=param_tunner['retorno_inicial']

        mapa={}
        paredes=gameState.getWalls() #Obtengo mapa de paredes
        for x in range(len(paredes.data)):
          for y in range(len(paredes.data[0])):#Recorro grid y obtengo mapa
            if paredes[x][y]:
              mapa[(x, y)] = '#'
            else:
              mapa[(x, y)] = ' '

        start=gameState.getGhostPosition(Agent)
        start=[round(start[0]),round(start[1])]
        end=gameState.getPacmanPosition()
        path = calcular_distancias.astar_search(mapa, start, end)#Uso algoritmo A* para calcular distancia
        #A* calcula distancia teniendo en cuenta obstaculos

        if gameState.getGhostState(Agent).scaredTimer>0:#Soy fantasma asustado
          retorno = retorno + param_tunner['ghost_pacman'] * len(path) #Penalizo según la distancia al Pacman
          retorno = - retorno
        else: #Soy fantasma comun
          retorno = retorno + param_tunner['ghost_pacman'] * len(path) #Penalizo según la distancia al Pacman

        if Is_End:
          if Pacman_Wins:
            retorno = -float('inf') #Pacman Gana, es peor estado
          else:
            retorno = float('inf') #Fantasma Gana, es el mejor estado
              
      retorno_vector.append(retorno)
    return retorno_vector

  # Función implementada para calcular la distancia entre 2 agentes
  # se utiliza en evaluationFunction
  def calcular_distancia(self, p1, p2):
    distancia = np.abs(p1[0]-p2[0])+np.abs(p1[1]-p2[1])
    return distancia

  def getAction(self, gameState):  
    action, value = self.maxN(gameState, self.index, self.max_depth)
    return action

  # Función recursiva que se encarga de maximizar el valor del estado
  # para el agente que esté jugando en ese momento. Esto se realiza
  # mientras no sea un caso base, ya sea que se alcanzó a profundidad
  # o que el juego haya terminado.
  def maxN(self, gameState: GameStateExtended, agentIndex, depth):  
    #Casos base:
    if depth == 0 and not gameState.isEnd():
      if self.unroll_type == "MC":
        values = self.montecarlo_eval(gameState, agentIndex)
        return None, values
      else:
        values = self.montecarlo_tree_search_eval(gameState, agentIndex)
        return None, values
    elif gameState.isEnd():
      #Paso info de que el juego es final y si pacmanes o no el ganador
      return None, self.evaluationFunction(gameState, agentIndex,1,gameState.isWin()) 
    
    #Llamada recursiva
    legalActions = gameState.getLegalActions(agentIndex)
    random.shuffle(legalActions)    
    nextAgent = self.getNextAgentIndex(agentIndex, gameState)
    action_nodes =[]
    for action in legalActions:
      child_node = gameState.deepCopy()
      nextState = child_node.generateSuccessor(agentIndex, action)
      _, state_value = self.maxN(nextState, nextAgent,depth-1)
      action_nodes.append((action, state_value))

    best_action = None
    best_score_array = np.zeros(gameState.getNumAgents())
    for action_node in action_nodes:

      if best_action == None:
        best_action=action_node[0]
        best_score_array=action_node[1]
      else:
        if action_node[1][agentIndex]>best_score_array[agentIndex]:
          best_action=action_node[0]
          best_score_array=action_node[1]
    
    return best_action, best_score_array
  

  #Esta función devuelve el siguiente agente
  # En el if se pregunta si es igual al agente +1, dado que getNumAgents 
  # devuelve el largo de la lista de agentes, y agentIndex es la posición 
  # del agente en la lista comenzando de 0
  def getNextAgentIndex(self, agentIndex, gameState):
    if gameState.getNumAgents() == agentIndex +1:
      return 0
    else:
      return agentIndex +1

  # En esta función se realizan los unrolls. 
  # Para eso evaluamos si se llegó a un estado final, o si
  # se alcanzó la cantidad de max_unroll_depth dado por parámetro
  # Si ninguna de estas condiciones se cumple, entonces se continuará
  # tomando una acción al azar, lo que implica avanzar en el juego.
  # En caso de que se cumpla la 2º condición se pasará a 
  # la función de evaluación.
  def random_unroll(self, gameState: GameStateExtended, agentIndex):
    done = gameState.isEnd()
    successor = gameState
    actual_unroll_depth = self.max_unroll_depth
    while not done and (actual_unroll_depth !=0):
      actions = successor.getLegalActions(agentIndex)
      action = random.choice(actions)
      successor = successor.generateSuccessor(agentIndex, action)
      agentIndex = self.getNextAgentIndex(agentIndex, successor)
      done = successor.isEnd() 
      actual_unroll_depth = actual_unroll_depth -1
    if done:
      return self.evaluationFunction(successor, agentIndex) #Se usa funcion de evaluación
    elif actual_unroll_depth == 0:
      return self.evaluationFunction(successor, agentIndex) #Se usa funcion de evaluación
    else:
      return np.zeros(gameState.getNumAgents())

  # Esta función va a calcular la función de evaluación al aplicar 
  # MoneCarlo, la cual consiste en aplicar random.unroll la cantidad
  # de veces que esté definido por parámetro, retornando un array
  # con los valores de unroll divido la cantidad de unrolls realizados.    
  def montecarlo_eval(self, gameState, agentIndex):
    values = np.zeros(gameState.getNumAgents())
    for _ in range(self.number_of_unrolls):
      unroll_values = self.random_unroll(gameState, agentIndex)
      values = np.add(values, unroll_values) 
    return np.true_divide(values, self.number_of_unrolls)  

  #Algoritmo de MCTS para realizar planificación el tiempo real
  #Al llegar a un nuevo estado, se elige la siguiente acción como resultado de una etapa de
  #planificación La política utilizada para balancear la exploración y la explotación fue UCB
  def montecarlo_tree_search_eval(self, gameState, agentIndex):
    
    root = mcts_util.MCTSNode(parent = None, action = None, player = agentIndex, numberOfAgents= gameState.getNumAgents())
    root, gameState = self.expansion_stage(root, gameState)
    best_reward = None
    best_action = None  
    
    for _ in range(self.number_of_unrolls):
      state = gameState.deepCopy()
      node = root
      sum_reward = 0
      #Etapa de selección:
      #Siguiendo la política del árbol (UCB) se elige el comienzo 
      #de una trayectoria desde la raíz hasta una hoja
      node_sel, gameState_sel = self.selection_stage(node, state)
      #Etapa de expansión:
      #El árbol es expandido desde el nodo hoja seleccionado agregando uno o más hijos 
      #(estados a los que se llega desde el estado hoja mediante acciones inexploradas)
      node_exp, gameState_exp = self.expansion_stage(node_sel, gameState_sel)
      #Etapa de simulación:
      #A partir del nodo hoja (o uno de sus nuevos hijos), se completa una trayectoria simulada
      #mediante la política rollout. En nuestro caso, utilizamos una política random mediante la función
      #random unroll
      sum_reward = self.random_unroll(gameState_exp, node_exp.player)
      #Etapa de backpropagation:
      #Con el retorno generado por la trayectoria simulada, se actualizan las estimaciones de los valores 
      #de los pares estado acción que están dentro del árbol
      self.back_prop_stage(node_exp, sum_reward) 
    index=0
    for child in root.children:
      #Si la raíz tiene hijos, devuelvo el hijo que tiene el mejor promedio en la posición del jugador de la raíz
      if index==0:
        best_action = child.action
        best_reward = np.true_divide(child.value, child.visits)
        index += 1
      else:
        valor_hijo = child.value[root.player]
        valor_best = best_reward[root.player]
        if valor_hijo > valor_best:
          best_action = child.action
          best_reward = np.true_divide(child.value, child.visits)     

    return best_reward
  
  #Etapa de selección del algoritmo MCTS
  def selection_stage(self, node, gameState):
    successor = gameState
    #Variable para ver si el estado es terminal
    done = successor.isEnd()
    #Tomo el estado actual
    while len(node.children)>0 and not done:
      #Exploro
      if node.explored_children < len(node.children):
        child = node.children[node.explored_children]
        node.explored_children += 1
        node = child
      else:
        #Exploto
        node = max(node.children, key= mcts_util.ucb)
      
      #Obtengo la acción
      action = node.action
      #Voy a estado siguiente
      successor = successor.generateSuccessor(node.parent.player, action)
    return node, successor

  #Expansión del MCTS desde el nodo hoja  
  def expansion_stage(self, node, gameState):
    if not gameState.isEnd():
      node.children = []
      for a in gameState.getLegalActions(node.player):
        parent = node
        action = a
        player = self.getNextAgentIndex(node.player,gameState)
        numberOfAgents = gameState.getNumAgents()
        nodo=mcts_util.MCTSNode(parent = parent, action = action, player = player, numberOfAgents = numberOfAgents)
        node.children.append(nodo)
      random.shuffle(node.children)
    return node, gameState

  #MCTS backprop para actualizar la información de los nodos
  def back_prop_stage(self, node, value):
    while node:
      node.visits += 1
      node.value = np.add(value,node.value)
      node = node.parent 
