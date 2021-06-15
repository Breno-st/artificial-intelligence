from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
import numpy as np
import random as random
import time

################ Functions for the evaluate #############

def evaluate_good_pieces(state, color):
    """
    Parameters
    ----------
    state : state of the game
    color : player we need informations on

    Returns
    -------
    corner : number of pieces that are in a corner
    edges : number of pieces that are near an edge (doesn t count corner pieces)
    """

    corner = 0
    edges = 0

    pieces = state.board.get_player_pieces_on_board(color)

    for position in pieces:
        if position==(0,0) or position==(4,4) or position==(0,4) or position==(4,0):
            corner = corner+1
        elif position[0]==0 or position[1]==0 or position[0]==4 or position[1]==4:
            edges = edges+1
    return corner, edges

def evaluate_direct_captures(state, opponent_color, allied_color):
    """
    Parameters
    ----------
    board : board of the game (given from a state)
    opponent_color : color of the opponent player
    allied_color : color of the allied player

    Returns
    -------
    threats : the maximum number of pieces the ennemy can capture in one move
    """

    pieces = state.board.get_player_pieces_on_board(opponent_color)

    threats = 0

    for position in pieces:

        moves = SeegaRules.get_effective_cell_moves(state, position)
        if moves:
            this_threats = 0
            for move in moves:
                this_piece = np.array(move)
                for new_piece in [this_piece+np.array([0,2]), this_piece+np.array([2,0]), this_piece-np.array([0,2]), this_piece-np.array([2,0])]:
                    in_between = (this_piece+new_piece)//2
                    if new_piece[0]>-1 and new_piece[0]<5 and new_piece[1]>-1 and new_piece[1]<5 and in_between[0]>-1 and in_between[0]<5 and in_between[0]>-1 and in_between[0]<5 and np.any(in_between!=[2, 2]):
                        if state.board.get_board_state()[new_piece[0]][new_piece[1]]==opponent_color and state.board.get_board_state()[in_between[0]][in_between[1]]==allied_color:
                            this_threats = this_threats+1
            if this_threats>threats:
                threats = this_threats
    return threats


def count_structure(piece, pieces, piece_list):

    for next_piece in pieces:
        if next_piece not in piece_list:
            if (next_piece[0]==piece[0] or next_piece[1]==piece[1]) and abs(next_piece[0]+next_piece[1]-piece[0]-piece[1])==1:
                piece_list.append(next_piece)
                count_structure(next_piece, pieces, piece_list)

def evaluate_structure(state, color):
    """

    Parameters
    ----------
    state : state of the game
    color : player we need informations on

    Returns
    -------
    biggest_structure : size of biggest structure
    """


    biggest_structure = 0
    piece_in_biggest_structure = []

    pieces = state.board.get_player_pieces_on_board(color)

    for piece in pieces:
        if piece not in piece_in_biggest_structure:
            piece_in_structure = [piece]
            count_structure(piece, pieces, piece_in_structure)
            if len(piece_in_structure)>biggest_structure:
                piece_in_biggest_structure = piece_in_structure
                biggest_structure = len(piece_in_structure)

    return biggest_structure

################ Functions for the successors #############

def total_threats(state, opponent_color, allied_color):
    """
    Parameters
    ----------
    board : board of the game (given from a state)
    opponent_color : color of the opponent player
    allied_color : color of the allied player

    Returns
    -------
    threats : the total number of pieces the ennemy can capture in one move (counts different moves)
    """

    pieces = state.board.get_player_pieces_on_board(opponent_color)

    threats = 0

    for position in pieces:

        moves = SeegaRules.get_effective_cell_moves(state, position)
        if moves:
            for move in moves:
                this_piece = np.array(move)
                for new_piece in [this_piece+np.array([0,2]), this_piece+np.array([2,0]), this_piece-np.array([0,2]), this_piece-np.array([2,0])]:
                    in_between = (this_piece+new_piece)//2
                    if new_piece[0]>-1 and new_piece[0]<5 and new_piece[1]>-1 and new_piece[1]<5 and in_between[0]>-1 and in_between[0]<5 and in_between[0]>-1 and in_between[0]<5 and np.any(in_between!=[2, 2]):
                        if state.board.get_board_state()[new_piece[0]][new_piece[1]]==opponent_color and state.board.get_board_state()[in_between[0]][in_between[1]]==allied_color:
                            threats = threats+1
    return threats


def evaluate_state_board(state, self_position):
    player_color = Color(self_position)
    # board_shape = (5, 5)
    board = state.board
    square = board.get_board_state()
    player_pieces = board.get_player_pieces_on_board(player_color)
    evaluation = 0
    for piece in player_pieces:
        y, x = piece[0], piece[1]
        evaluation = (evaluation+1) if (y+1<5 and (square[x][y+1].value==0 or square[x][y+1].value==self_position)) else (evaluation)
        evaluation = (evaluation+1) if (x+1<5 and (square[x+1][y].value==0 or square[x+1][y].value==self_position)) else (evaluation)
        evaluation = (evaluation+1) if (y-1<5 and (square[x][y-1].value==0 or square[x][y-1].value==self_position)) else (evaluation)
        evaluation = (evaluation+1) if (x-1<5 and (square[x-1][y].value==0 or square[x-1][y].value==self_position)) else (evaluation)

    return evaluation

def defensive_evaluation(self_position, state):

    if state.phase == 1: # If we are in the placement phase, we return 0
        # This score will lead the algorithm to form a barrier
        return 0
    else:

        agent_score = state.get_player_info(self_position)["score"]
        enemy_score = state.get_player_info(-self_position)["score"]
        dif = agent_score - enemy_score

        sfty_coef = (1/8)  # A coef to stablish how much the safety will be considered

        if SeegaRules.is_end_game(state) and dif < 0: # If it is a state where the agent lose
            return float('-inf')
        elif SeegaRules.is_end_game(state) and dif > 0: # If it is a state where the agent win
            return float('inf')
        else: # It is an intermediary state
            return dif + sfty_coef * evaluate_state_board(state, self_position)

def sort_successors(succs, state, allied_color, opponent_color):
    """
    Parameters
    ----------
    succs : list of successors as return by the AI successors function
    state : current state of which succs is the list of successors
    allied_color : color of allied player
    opponent_color : color of opponent player
    Returns
    -------
    to_return : a sorted list of successors in the following way. First the aggressive moves (moves that capture opponent pieces).
    Then the defensive moves (moves that reduce the number of allied pieces the opponent can take).
    Then boring moves (the rest of the moves).
    Also each of the three intermediate lists are randomly shuffled (see AI book page 169 as to why)
    """

    aggressive_moves = []
    defensive_moves = []
    boring_moves = []

    allied_pieces = len(state.board.get_player_pieces_on_board(allied_color))
    opponent_pieces = len(state.board.get_player_pieces_on_board(opponent_color))
    threats = total_threats(state, opponent_color, allied_color)

    for (action, new_state) in succs:
        if len(new_state.board.get_player_pieces_on_board(opponent_color))<opponent_pieces:
            aggressive_moves.append((action, new_state))
        elif total_threats(new_state, opponent_color, allied_color)<threats:
            defensive_moves.append((action, new_state))
        else:
            boring_moves.append((action, new_state))

    #Shuffle all 3 lists
    random.shuffle(aggressive_moves)
    random.shuffle(defensive_moves)
    random.shuffle(boring_moves)

    to_return = []

    to_return.extend(defensive_moves)
    to_return.extend(aggressive_moves)

    if len(aggressive_moves)==0 and len(defensive_moves)==0: # if no aggressive moves nor defensive moves return boring moves
        to_return.extend(boring_moves)

    elif allied_pieces>opponent_pieces+1: # If we have at least 2 more pieces than our adversary, take boring moves into account as boring game will result in a win
        to_return.extend(boring_moves)

    return to_return

################ Functions for the cutoff #############

def calculate_time(remain_time):

    return min(np.exp(0.03*remain_time)-1, 20) # If 60 seconds remaining, return 5
    #return min(np.exp(0.04*remain_time)-1, 20) # If 60 seconds remaining, return 10
    #return min(np.exp(0.05*remain_time-1), 20) # If 60 seconds remaining, return 19

################ Class Player #############

class AI(Player):

    in_hand = 12
    score = 0
    name = "smart_v4"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.flat_max_depth = 0

    def play(self, state, remain_time):
        print("")
        print(f"Player {self.position} is playing.")
        print("time remain is ", remain_time, " seconds")
        print("smart agent playing in phsae", state.phase)

        if len(self.successors(state))==1: # If there is only one successor, no need to do computations
            return self.successors(state)[0][0]

        self.state_dict = {}

        self.allocated_time = calculate_time(remain_time)
        self.start_turn = time.time()

        if state.phase==1:
            return minimax_search(state, self)
        else:
            self.flat_max_depth = 2
            last_iteration_time = 0

            time_remaining_this_turn = self.allocated_time-(time.time()-self.start_turn)

            while last_iteration_time<time_remaining_this_turn and self.flat_max_depth<10:
                start_iteration = time.time()
                new_action = minimax_search(state, self)
                if new_action != None:
                    action = new_action
                end_iteration = time.time()
                last_iteration_time = end_iteration-start_iteration
                self.flat_max_depth += 1
                self.state_dict = {}
                time_remaining_this_turn = self.allocated_time-(time.time()-self.start_turn)

            print(self.flat_max_depth, time_remaining_this_turn, self.allocated_time, time.time())

            return action

    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """

    def successors(self, state):

        next_player = state.get_next_player()

        actions = SeegaRules.get_player_actions(state, next_player)
        succs = []
        for action in actions:
            state_copy = deepcopy(state)
            resulting_state = SeegaRules.act(state_copy, action, next_player)
            if resulting_state is not False:
                succs.append((action, resulting_state[0]))

        if state.phase==2:
            succs.sort(key=lambda x: self.evaluate(x[1]), reverse=next_player != self.position)
        #else:
        #    random.shuffle(succs) # All moves will be considered as boring moves so no need to do computations here

        return succs

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):

        if state in self.state_dict and self.state_dict[state]==[state.score[-1], state.score[1]]: # Redundant state
            return True
        else:
            self.state_dict[state] = [state.score[-1], state.score[1]]

        if SeegaRules.is_end_game(state):
            return True
        else:
            if state.phase==1 and depth>0:
                return True
            if depth>self.flat_max_depth:
                return True
            else:
                if time.time()-self.start_turn>self.allocated_time:
                    return True
                else:
                    return False

    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):

        evaluate_score = 0
        if state.phase==2:
            opponent_color = Color(-self.position)
            allied_color = Color(self.position)

            # Get the defensive_evaluation
            def_eval = defensive_evaluation(self.position, state)

            # #Get number of allied and opponent pieces
            opponent_pieces = len(state.board.get_player_pieces_on_board(opponent_color))
            allied_pieces = len(state.board.get_player_pieces_on_board(allied_color))

            # #get number of allied and opponent corner/edge pieces (more valuable pieces)
            allied_corner, allied_edges = evaluate_good_pieces(state, allied_color)
            opponent_corner, opponent_edges = evaluate_good_pieces(state, opponent_color)

            #get number of direct captures opponent can make
            direct_captures = evaluate_direct_captures(state, opponent_color, allied_color)

            #get biggest connected structures
            allied_structure = evaluate_structure(state, allied_color)
            opponent_structure = evaluate_structure(state, allied_color)

            evaluate_score += (allied_pieces-opponent_pieces)*5 # Difference in pieces
            evaluate_score += allied_corner*2 # Count corner pieces 3 times in total as they are uncapturable
            evaluate_score += allied_edges # Count edge pieces 2 times in total as they are more difficult to capture than center pieces but still capturable
            evaluate_score -= opponent_edges # Decrease for opponent edge pieces as if we can capture some it is very good. No decrease for corner pieces as we can not capture them
            evaluate_score -= direct_captures # Decrease for direct captures the opponent can make
            evaluate_score += allied_structure # Increase for the size of our biggest structure
            evaluate_score -= opponent_structure # Decrease for the size of the opponent structure
            evaluate_score += 2 * def_eval

        return evaluate_score

    """
    Specific methods for a Seega player (do not modify)
    """
    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos['in_hand']
        self.score = infos['score']

    def reset_player_informations(self):
        self.in_hand = 12
        self.score = 0



"""
MiniMax and AlphaBeta algorithms.
Adapted from:
    Author: Cyrille Dejemeppe <cyrille.dejemeppe@uclouvain.be>
    Copyright (C) 2014, Universite catholique de Louvain
    GNU General Public License <http://www.gnu.org/licenses/>
"""

inf = float("inf")

def minimax_search(state, player, prune=True):
    """Perform a MiniMax/AlphaBeta search and return the best action.

    Arguments:
    state -- initial state
    player -- a concrete instance of class AI implementing an Alpha-Beta player
    prune -- whether to use AlphaBeta pruning

    """
    def max_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        val = -inf
        action = None
        for a, s in player.successors(state):
            if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                v, _ = max_value(s, alpha, beta, depth + 1)
            else:                                             # next turn is for the other one
                v, _ = min_value(s, alpha, beta, depth + 1)
            if v > val:
                val = v
                action = a
                if prune:
                    if v >= beta:
                        return v, a
                    alpha = max(alpha, v)
        return val, action

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        val = inf
        action = None
        for a, s in player.successors(state):
            if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                v, _ = min_value(s, alpha, beta, depth + 1)
            else:                                             # next turn is for the other one
                v, _ = max_value(s, alpha, beta, depth + 1)
            if v < val:
                val = v
                action = a
                if prune:
                    if v <= alpha:
                        return v, a
                    beta = min(beta, v)
        return val, action

    _, action = max_value(state, -inf, inf, 0)
    return action
