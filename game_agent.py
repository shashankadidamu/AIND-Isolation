"""This file contains all the classes you must complete for this project.
You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.
You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

"""For Minimax and alphabeta used psuedocode present in 
    https://en.wikipedia.org/wiki/Minimax
    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
"""
import math
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    moves_player1 = len(game.get_legal_moves(player))
    moves_player2 = len(game.get_legal_moves(game.get_opponent(player)))
    return float(moves_player1-2*moves_player2)

    empty_cells = len(game.get_blank_spaces())
    
    #heuristic
    #return float(moves_player1-(empty_cells*moves_player2))


    # heuristic
    #player1_row_location, player1_column_location = game.get_player_location(player)
    #centerColumn = int(round(game.height/2))
    #centerRow = int(round(game.width/2))

    #dist_from_center = math.sqrt((player1_column_location-centerColumn)**2+(player1_row_location-centerRow)**2)

    #player2_row_location, player2_column_location = game.get_player_location(game.get_opponent(player))

    #player2_dist_from_center = math.sqrt((player2_column_location-centerColumn)**2+(player2_row_location-centerRow)**2)

    #if dist_from_center > player2_dist_from_center:
    #    return float((moves_player1-dist_from_center)-moves_player2)
    #else:
    #    return float(moves_player1+dist_from_center-(moves_player2+player2_dist_from_center))





class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).
    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.
        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!
        if len(legal_moves)==0:
            return(-1,-1)

        legal_move = legal_moves[random.randint(0,len(legal_moves)-1)]
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            #iterative deepening
            if self.iterative:
                depth = 0
                while True:
                    depth += 1
                    if self.method == 'alphabeta':
                        score,legal_move = self.alphabeta(game,depth)                     
                    else:
                        score,legal_move = self.minimax(game,depth)

            #fixed depth search 
            else:
                if self.method == 'alphabeta':
                    score,legal_move =  self.alphabeta(game,self.search_depth)
                else:
                    score,legal_move =  self.minimax(game,self.search_depth)


        except Timeout:
            # Handle any actions required at timeout, if necessary
            return legal_move

        # Return the best move from the last completed search iteration
        return legal_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        bestMove = (-1,-1)
        if maximizing_player:
            current_score = float("-inf")
        else:
            current_score = float("inf")

        if depth == 0:
            return self.score(game,self),game.get_player_location(self)  
        
        possible_legal_moves = game.get_legal_moves()
        for current_move in possible_legal_moves:
            next_move = game.forecast_move(current_move)

            if maximizing_player:
                score,_=self.minimax(next_move,depth-1,False)
                if score>current_score:
                    current_score = score
                    bestMove = current_move

            else:
                score,_=self.minimax(next_move,depth-1,True)
                if score<current_score:
                    current_score=score
                    bestMove=current_move

        return current_score,bestMove



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        bestMove = (-1,-1)
        if maximizing_player:
            current_score = float("-inf")
        else:
            current_score = float("inf")

        if depth == 0:
            return self.score(game,self),game.get_player_location(self)  
        
        possible_legal_moves = game.get_legal_moves()
        for current_move in possible_legal_moves:
            next_move = game.forecast_move(current_move)

            if maximizing_player:
                score,_ = self.alphabeta(next_move,depth-1,alpha,beta,False)
                if score>current_score:
                    current_score=score
                    bestMove=current_move

                if current_score>=beta:
                    return current_score,bestMove #prunning

                if current_score>alpha:
                    alpha=current_score

            else:
                score,_=self.alphabeta(next_move,depth-1,alpha,beta,True)
                if score<current_score:
                    current_score=score
                    bestMove=current_move

                if current_score<=alpha:
                    return current_score,bestMove  #prunning

                if current_score<beta:
                    beta = current_score

        return current_score,bestMove #Final score