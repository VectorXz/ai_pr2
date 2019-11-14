"""
This module contains agents that play reversi.

Version 3.0
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2



_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color
    
    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(
                target=self.search, 
                args=(
                    self._color, board, valid_actions, 
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""
    
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


"""
Default Zone of Sunat Agent 
"""


class SunatAgent(ReversiAgent):  # Create Sunat Agent use Alpha-Beta Pruning Search

    """
    Alpha-Beta Pruning Search Algorithm limit with 10
    -----------------------------------------------
    function ABS(state) return action
    v <- Max_value(state,+inf,-inf) # v = Max_Value
    -----------------------------------------------
    function Max_value(state,alpha,beta) return a utility value
    if Terminal-test(state) then return utility(state)
    v <- (-inf)  # we can define -inf as float('-inf') or -float('inf)
    for each a in action(state) do
    v <- max(v,Min_Value(Result(s,a),alpha,beta))
    if v >= beta then return v
    alpha <- max(alpha,v)
    return v
    -----------------------------------------------
    function Min_value(state,alpha,beta) return a utility value
    if Terminal-test(state) then return utility(state)
    v <- (inf)
    for each a in action(state) do
    v <- min(v,Max_Value(Result(s,a),alpha,beta))
    if v <= alpha then return v
    beta <- max(beta,v)
    return v
    -----------------------------------------------
    function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            α := max(α, value)
            if α ≥ β then
                break (* β cut-off *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            β := min(β, value)
            if α ≥ β then
                break (* α cut-off *)
        return value
    (* Initial call *)
    alphabeta(origin, depth, −∞, +∞, TRUE)
    """

    def __index__(self):
        super(SunatAgent, self)

    def search(self, color, board, valid_actions, output_move_row, output_move_column):  # Instead Alpha Beta Search
        # time.sleep(0.75)
        print(" Sunat AI is Thinking...")
        try:
            # if self._color == 1 :
            # if condition is does not need because we found color player always 1
            # print("Sunat is making the decision")
            # best_state = self.minmax(board,len(valid_actions)-1,3,0, float('-inf'),float('inf'),True)
            # moving = self.Max_value(board,valid_actions,4,0,MIN,MAX,True)
            evaluation, best_state = self.Max_value(board, valid_actions, 5, 0, -float('inf'), float('inf'), True)
            # we found depth level between 1 - 4 is found solution quicker
            print(" Choice A")
            # Sunat_Action = valid_actions[best_state]
            # output_move_row.value = Sunat_Action[0]
            # output_move_column.value = Sunat_Action[1]
            # else:
            # print("Sunat is making the decision")
            # evaluation, best_state = self.Max_value(board,valid_actions,2,0,-float('inf'),float('inf'),True)
            # print(" Choice B")
            # Sunat_Action = valid_actions[best_state]
            # output_move_row.value = Sunat_Action[0]
            # output_move_column.value = Sunat_Action[1]
            # Sunat_Action = valid_actions[moving]

            # print(" Sunat Selected:" + str(best_state))
            # time.sleep(0.25)
            print(" Sunat is making the decision")
            # time.sleep(0.25)
            output_move_row.value = best_state[0]
            output_move_column.value = best_state[1]
            print(" Sunat Selected:" + str(best_state))
            # time.sleep(1.25)
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    # def MinimaxABP(self,board:np.array,validactions:np.array,depth:int,level:int,alpha:float,beta:float,gain:bool):
    #     if depth == 0:
    #         return self.evaluateStatistically(board)
    #     if self._color == 1:
    #         value = self.Max_value(board,validactions,4,0,-10000.0,10000.0,True)
    #     else:
    #         value = self.Max_value(board,validactions,2,0,-10000.0,10000.0,True)
    #         # best_action: np.array = None
    #     # alpha = -0x40000
    #     # beta = 0x40000
    #     best_state: np.array = None
    def Max_value(self, board: np.array, validactions: np.array, depth: int, level: int, alpha: float, beta: float,
                  gain: bool):
        if depth == 0:
            countA: int = 0
            countB: int = 0
            evalBoard = np.array(list(zip(*board.nonzero())))

            for row in evalBoard:
                if board[row[0]][row[1]] == self._color:
                    countA += 1
                else:
                    countB += 1
            return countA - countB

        best_state: np.array = None
        MaxAlpha: int = alpha
        Maxevaluation = -float('inf')
        player: int = self._color
        for a in validactions:
            newstate, newaction = self.createState(board, a, player)
            newstep = self.Min_value(newstate, newaction, depth - 1, level + 1, MaxAlpha, beta, not gain)
            if Maxevaluation < newstep:
                Maxevaluation = newstep

                if level == 0:
                    best_state = a

            MaxAlpha = max(MaxAlpha, Maxevaluation)
            if beta <= MaxAlpha:
                break
        if level != 0:
            return Maxevaluation
        else:
            return Maxevaluation, best_state

    def Min_value(self, board: np.array, validactions: np.array, depth: int, level: int, alpha: float, beta: float,
                  gain: bool):
        if depth == 0:
            countA: int = 0
            countB: int = 0
            evalBoard = np.array(list(zip(*board.nonzero())))

            for row in evalBoard:
                if board[row[0]][row[1]] == self._color:
                    countA += 1
                else:
                    countB += 1
            return countA - countB

        MinBeta: int = beta
        Minevaluation = float('inf')
        player: int = self.getOpponent(self._color)
        best_state: np.array = None
        for a in validactions:
            newstate, newaction = self.createState(board, a, player)
            newstep = self.Max_value(newstate, newaction, depth - 1, level + 1, alpha, MinBeta, not gain)

            if Minevaluation > newstep:
                Minevaluation = newstep

                if level == 0:
                    best_state = a

            MinBeta = min(MinBeta, newstep)
            if MinBeta <= alpha:
                break
        if level != 0:
            return Minevaluation
        else:
            return Minevaluation, best_state
            # self._move = best_state[value]

    # def evaluateStatistically(self, board: np.array) -> int:
    #     countA: int = 0
    #     countB: int = 0
    #     evalBoard = np.array(list(zip(*board.nonzero())))

    #     # print("Print Board: " + str(evalBoard))
    #     for row in evalBoard:
    #         if board[row[0]][row[1]] == self._color:
    #             countA += 1
    #         else:
    #             countB += 1
    #     return countA - countB

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def createState(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        newState: np.array = transition(board, player, action)

        validMoves: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves: np.array = np.array(list(zip(*validMoves.nonzero())))

        return newState, validMoves


"""
End Zone of Sunat Agent
"""
