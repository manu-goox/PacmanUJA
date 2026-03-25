# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = util.Stack()
    frontier.push((start, []))  # (state, path_actions)
    explored = set()

    while not frontier.isEmpty():
        state, actions = frontier.pop()

        if state in explored:
            continue
        explored.add(state)

        if problem.isGoalState(state):
            return actions

        for succ, action, stepCost in problem.getSuccessors(state):
            if succ not in explored:
                frontier.push((succ, actions + [action]))

    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = util.PriorityQueue()

    frontier.push((start, [], 0), heuristic(start, problem))

    best_g = {start: 0}

    while not frontier.isEmpty():
        state, actions, g = frontier.pop()

        if g > best_g.get(state, float("inf")):
            continue

        if problem.isGoalState(state):
            return actions

        for succ, action, stepCost in problem.getSuccessors(state):
            new_g = g + stepCost
            if new_g < best_g.get(succ, float("inf")):
                best_g[succ] = new_g
                new_actions = actions + [action]
                f = new_g + heuristic(succ, problem)
                frontier.push((succ, new_actions, new_g), f)

    return []


def exploration(problem):
    """
    Exploración completa del laberinto mediante DFS con backtracking explícito.

    El agente avanza hacia celdas no visitadas y, cuando no encuentra ninguna,
    retrocede paso a paso hasta encontrar un camino nuevo. Esto garantiza que
    el plan de acciones resultante pase físicamente por todas las celdas
    alcanzables del laberinto.

    Retorna una lista de acciones (Directions) que incluye tanto avances
    como retrocesos, cubriendo el máximo de casillas posible.
    """

    # Mapa de dirección opuesta para poder retroceder
    opposite = {
        Directions.NORTH: Directions.SOUTH,
        Directions.SOUTH: Directions.NORTH,
        Directions.EAST:  Directions.WEST,
        Directions.WEST:  Directions.EAST,
    }

    # Casilla Inicial
    start = problem.getStartState() 

    # Creamos un set con las casillas exploradas y añadimos la inicial
    visited = set()
    visited.add(start)

    # Lista de acciones realizadas
    actions = []

    # Pila del camino actual: lista de acciones tomadas
    path_stack = []  

    # Variable para saber la posicion actual
    current = start

    while True:
        # Buscamos un sucesor no visitado desde la celda actual
        successors = problem.getSuccessors(current)
        next_move = None

        # Bucle para encontrar la siguiente celda adyacente no visitada
        for next_state, action, _ in successors:
            if next_state not in visited:
                next_move = (next_state, action)
                break

        if next_move is not None:

            # Hay celda nueva por tanto, avanzamos hacia ella
            next_state, action = next_move

            # Añadimos a las casillas visitadas la siguiente celda
            visited.add(next_state)

            # Añadimos a las acciones realizadas
            actions.append(action)
            path_stack.append(action)

            # Actualizamo la posicion actual
            current = next_state

        else:
            if not path_stack:
                break
            
            # No hay posibles movimientos asi que retrocederemos una accion
            last_action = path_stack.pop()
            back_action = opposite[last_action]

            # Añadimos la accion de retroceder a las acciones realizadas
            actions.append(back_action)
            
            # Recalculamos la posición actual retrocediendo
            dx, dy = {'North': (0,1), 'South': (0,-1), 'East': (1,0), 'West': (-1,0)}[back_action]
            x, y = current
            current = (int(x + dx), int(y + dy))           

    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
exp = exploration
