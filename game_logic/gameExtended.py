from .game import Game, Directions, GameStateData, AgentState, Configuration
from .pacman import readCommand, ClassicGameRules, GameState, GhostRules, PacmanRules, TIME_PENALTY
import numpy as np

class GameExtended(Game):
    def init(self):
        # try:
        #     import boinc
        #     _BOINC_ENABLED = True
        # except:
        _BOINC_ENABLED = False

        """
    Main control loop for game play.
    """
        self.display.initialize(self.state.data)
        self.numMoves = 0

        # self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                # this is a null agent, meaning it failed to load
                # the other team wins
                self._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.mute()
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(
                            agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return
                    except Exception as data:
                        self.unmute()
                        self._agentCrash(i, quiet=True)
                        return
                else:
                    agent.registerInitialState(self.state.deepCopy())
                # TODO: could this exceed the total time
                self.unmute()

    def step(self, action, agentIndex, render = None):
        _BOINC_ENABLED = False
        #agentIndex = self.startingIndex
        numAgents = len(self.agents)
        assert(not self.gameOver and agentIndex < numAgents)
        
        # Fetch the next agent
        agent = self.agents[agentIndex]
        move_time = 0
        skip_action = False
        # Generate an observation of the state
        if False and 'observationFunction' in dir(agent):
            pass                
        else:
            observation = self.state.deepCopy()

        # Execute the action
        self.moveHistory.append((agentIndex, action))
        if False and self.catchExceptions:
            pass                
        else:
            self.state = self.state.generateSuccessor(agentIndex, action)

        # Change the display
        ###idx = agentIndex - agentIndex % 2 + 1
        ###self.display.update( self.state.makeObservation(idx).data )

        # Allow for game specific conditions (winning, losing, etc.)
        self.rules.process(self.state, self)
        # Track progress
        if agentIndex == numAgents + 1:
            self.numMoves += 1
        
        if _BOINC_ENABLED:
            boinc.set_fraction_done(self.getProgress())
        
        self.render()
        
        if self.gameOver:
            self.display.finish()

        return self.state

    def render(self):
        self.display.update(self.state.data)

    def close(self):
        self.display.finish()


class ClassicGameRulesExtended(ClassicGameRules):
    def newGame(self, layout, pacmanAgent, ghostAgents, display, quiet=False, catchExceptions=False):        
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameStateExtended()
        initState.initialize(layout, len(ghostAgents))
        game = GameExtended(agents, display, self,
                            catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game


class GameStateExtended(GameState):   
    def __init__( self, prevState = None ):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState is not None: # Initial state
            self.data = GameStateDataExtended(prevState.data)
        else:
            self.data = GameStateDataExtended()
    
    def get_rewards(self):
        if self.isEnd():
            rewards =  []
            for agentIndex in range(self._get_num_agents()):
                rewards.append(self._get_agent_reward(agentIndex))
        else:
            rewards = list(np.zeros(self._get_num_agents()))
        return rewards

    def _get_agent_reward(self, agentIndex):
        if self.isEnd():
            if agentIndex == 0 and not self.data._win:
                extra = -1000
            elif agentIndex != 0 and self._check_ghost_has_eaten_pacman(agentIndex):
                extra = +1000
            else:
                extra = 0
            reward = self.data.scores[agentIndex] + extra
        else: 
            reward = 0
        return reward

    def _check_ghost_has_eaten_pacman(self, agentIndex):
        if agentIndex == 0 or agentIndex >= self._get_num_agents():
            raise Exception('Agent index is not from a Ghost')
        agent_pos = self.data.agentStates[agentIndex].configuration.pos
        agent_pos = int(agent_pos[0]), int(agent_pos[1])
        pacman_pos = self.data.agentStates[0].configuration.pos
        pacman_pos = int(pacman_pos[0]), int(pacman_pos[1])
        
        return agent_pos == pacman_pos

    def _get_num_agents(self):
        return len(self.data.agentStates)

    def isEnd(self):
        return self.isWin() or self.isLose() or self.getScore() < -1500
    
    def deepCopy( self ):
        state = GameStateExtended( self )
        state.data = self.data.deepCopy()
        return state
    
    def generateSuccessor( self, agentIndex, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose(): raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = self.deepCopy()

        # Let agent's logic deal with its action's effects on the board
        if agentIndex == 0:  # Pacman is moving
            state.data._eaten = [False for i in range(state.getNumAgents())]
            PacmanRules.applyAction( state, action )
        else:                # A ghost is moving
            GhostRules.applyAction( state, action, agentIndex )

        # Time passes
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY # Penalty for waiting around
        else:
            GhostRules.decrementTimer( state.data.agentStates[agentIndex] )

        # Resolve multi-agent effects
        GhostRules.checkDeath( state, agentIndex )

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
                
        
        state.data.scores[0] = state.data.score
        if agentIndex != 0:
            state.data.scores[agentIndex] += -1

        assert(state.data.scores[0]==state.data.score)
        return state

class GameStateDataExtended(GameStateData):    
    def __init__( self, prevState = None ):        
        super().__init__(prevState)
        if prevState != None:
            self.scores = np.copy(prevState.scores)
        
    def initialize( self, layout, numGhostAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.copy()
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.scoreChange = 0
        self.agentStates = []
        numGhosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if numGhosts == numGhostAgents: continue # Max ghosts reached already
                else: numGhosts += 1
            self.agentStates.append( AgentState( Configuration( pos, Directions.STOP), isPacman) )
        
        self.scores = list(np.zeros(len(self.agentStates)))
        self._eaten = [False for a in self.agentStates]

    def deepCopy( self ):
        state = GameStateDataExtended( self )
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._capsuleEaten = self._capsuleEaten
        state.scores = self.scores
        return state