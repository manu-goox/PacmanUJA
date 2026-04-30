from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState
from collections import deque



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):

    if currentGameState.isWin():
        return 1e6
    if currentGameState.isLose():
        return -1e6

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # COMIDA (muy importante)
    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPos, f) for f in foodList)
        score += 12.0 / (minFoodDistance + 1.0)
        score -= 4.0 * len(foodList)

    # FANTASMAS
    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            score += 4.0 / (dist + 1.0)
        else:
            if dist <= 1:
                score -= 200
            else:
                score -= 2.5 / dist

    return score


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Agente MiniMax con profundidad limitada.
      Pacman (agente 0) maximiza y el fantasma (agente 1) minimiza.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    	super().__init__(evalFn,depth)
    	self.__numMovimientos = 0


    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Don't forget to limit the search depth using self.depth. Also, avoid modifying
          self.depth directly (e.g., when implementing depth-limited search) since it
          is a member variable that should stay fixed throughout runtime.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE
        # Contador de decisiones tomadas por Pacman (una por llamada a getAction)
        self.__numMovimientos += 1

        def minimax(state: GameState, depth: int, agentIndex: int) -> float:
            """
            Evalúa recursivamente un estado con MiniMax.
            - state: estado actual del juego.
            - depth: número de turnos completos de Pacman explorados.
            - agentIndex: agente que mueve en este nodo (0=MAX, 1=MIN).
            """
            # Parada por estado terminal: en terminales se usa score real del entorno.
            if state.isWin() or state.isLose():
                return state.getScore()
                

            # Parada por profundidad límite: en no terminales se usa evaluación heurística.
            if depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            
            # Sin acciones legales, el nodo se considera final para la búsqueda.
            if not legalActions:
                return state.getScore()

            # Nodo MAX: Pacman elige la acción con mayor valor.
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    bestValue = max(bestValue, minimax(successor, depth, 1))
                return bestValue

            # Nodo MIN: fantasma elige el menor valor.
            # depth solo aumenta al volver a Pacman porque eso marca una ronda completa.
            bestValue = float('inf')
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                bestValue = min(bestValue, minimax(successor, depth + 1, 0))
            return bestValue

        # En la raíz, Pacman selecciona la mejor acción según el valor MiniMax.
        legalPacmanActions = gameState.getLegalActions(0)
        # Evitar quedarse parado
        if Directions.STOP in legalPacmanActions:
            legalPacmanActions.remove(Directions.STOP)
        if not legalPacmanActions:
            return Directions.STOP

        bestActions = []
        bestValue = float('-inf')

        
        reverse = Directions.REVERSE[gameState.getPacmanState().getDirection()]

        for action in legalPacmanActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)

            # Penalizar volver atrás
            if action == reverse:
                value -= 2  # penalización ligera

            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        chosen = random.choice(bestActions)

        # Miramos si el siguiente estado termina el juego
        nextState = gameState.generateSuccessor(0, chosen)

        if nextState.isWin() or nextState.isLose():
            print("Movimientos totales:", self.__numMovimientos)

        return chosen
        # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
      You may reference the pseudocode for Alpha-Beta pruning here:
      en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    	super().__init__(evalFn,depth)
    	self.__numMovimientos = 0
    	
    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 43 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Agente ExpectiMax con profundidad limitada.
    Pacman (agente 0) maximiza y el fantasma (agente 1)
    se modela como un agente estocástico (azar uniforme).
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.__numMovimientos = 0

        # 🔴 MEMORIA ANTI-BUCLE
        self.recentPositions = deque(maxlen=6)
        self.lastAction = None

    def getAction(self, gameState: GameState) -> str:

        self.__numMovimientos += 1

        def expectimax(state: GameState, depth: int, agentIndex: int) -> float:

            if state.isWin() or state.isLose():
                return state.getScore()

            if depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return state.getScore()

            # MAX (Pacman)
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    bestValue = max(bestValue, expectimax(successor, depth, 1))
                return bestValue

            # AZAR (fantasma)
            total = 0.0
            probability = 1.0 / len(legalActions)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                total += probability * expectimax(successor, depth + 1, 0)

            return total

        # 🔴 ACCIONES LEGALES
        legalPacmanActions = gameState.getLegalActions(0)

        # ❌ eliminar STOP (muy importante)
        if Directions.STOP in legalPacmanActions:
            legalPacmanActions.remove(Directions.STOP)

        if not legalPacmanActions:
            return Directions.STOP

        bestValue = float('-inf')
        bestActions = []

        reverse = Directions.REVERSE[gameState.getPacmanState().getDirection()]

        for action in legalPacmanActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)

            nextPos = successor.getPacmanPosition()

            # 🔴 penalizar bucles
            if nextPos in self.recentPositions:
                value -= 3

            # 🔴 penalizar reversa
            if action == reverse:
                value -= 1

            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        chosen = random.choice(bestActions)

        # 🔴 guardar historial
        self.recentPositions.append(gameState.getPacmanPosition())
        self.lastAction = chosen

        # Miramos si el estado actual termina el juego
        nextState = gameState.generateSuccessor(0, chosen)

        if nextState.isWin() or nextState.isLose():
            print("Movimientos totales:", self.__numMovimientos)

        return chosen


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

    # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


# Abbreviation
better = betterEvaluationFunction



