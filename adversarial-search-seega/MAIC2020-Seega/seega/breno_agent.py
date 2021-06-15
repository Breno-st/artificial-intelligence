from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
from seega import SeegaAction
import random
import time
import numpy as np

######################
# Auxiliars Functions
######################
def calculate_time(remain_time):
    return min(np.exp(0.03*remain_time)-1, 20)

"""
Return number of opponent' possible captures
for given state
"""
def list_piece_connections(piece, all_pieces, piece_list):

    for next_piece in all_pieces:
        if next_piece not in piece_list:
            # check if they are in same row or column
            if (next_piece[0]==piece[0] or next_piece[1]==piece[1]):
                # check if they are close
                if abs(next_piece[0] + next_piece[1] - piece[0] - piece[1])==1:
                    piece_list.append(next_piece)
                    list_piece_connections(next_piece, all_pieces, piece_list)


###########
# AI Class
###########

class AI(Player):

    in_hand = 12
    score = 0
    name = "smart_agent"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.depth = 0


    """
    How our agent will play at each new state and remaining
    time.
    """
    def play(self, state, remain_time):
        # print("")
        # print(f"Player {self.position} is playing.")
        # print("time remain is ", remain_time, " seconds")

        self.state_dict = {}
        self.remaining_time = calculate_time(remain_time)
        self.start_time = time.time()

        if len(self.successors(state))==1:
            return self.successors(state)[0][0]

        if state.phase == 1: # add
            return minimax_search(state, self)
        else:
            self.depth = 2
            time_lapse = 0

            turn_remaining_time = self.remaining_time - (time.time() - self.start_time)

            while time_lapse < turn_remaining_time and self.depth < 10:
                start_iteration = time.time()
                new_action = minimax_search(state, self)
                if new_action != None:
                    action = new_action
                end_iteration = time.time()
                time_lapse = end_iteration - start_iteration
                self.depth += 1
                self.state_dict = {}
                turn_remaining_time = self.remaining_time - (time.time()- self.start_time)

            # print(self.depth, turn_remaining_time, self.remaining_time)

            return action


    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        # #player = state._next_player
        # actions = SeegaRules.get_player_actions(state, self.color.value)
        # SeegaRules.act(s, a, self.color.value)

        next_player = state._next_player
        actions = SeegaRules.get_player_actions(state, next_player)
        successors = list()

        for a in actions:
            s = deepcopy(state)
            possible_states = SeegaRules.act(s, a, next_player)
            if possible_states:
                successors.append((a,possible_states[0]))

        if state.phase == 2:
            successors.sort(key=lambda t: self.evaluate(t[1]), reverse=next_player != self.position)

        return successors

    """
    ** The cutoff function returns true if the alpha-beta/minimax
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
            if depth > self.depth:
                return True
            else:
                if time.time() - self.start_time > self.remaining_time:
                    return True
                else:
                    return False

    """
    Return a value correspond to number of
    harmless squares
    """
    def safety_evaluation(self, state):
        # get board info from state
        dimension = state.board.board_shape
        square = state.board.get_board_state()
        #player_color = Color(self.position)

        # initiate verification
        evaluation = 0
        pieces_on_board = state.board.get_player_pieces_on_board(Color(self.position))

        for piece in pieces_on_board:
            moves = [(piece[0] + a[0], piece[1] + a[1]) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            if (0 <= piece[0] + a[0] < dimension[0]) and (0 <= piece[1] + a[1] < dimension[1])]

            for move in moves:
                if square[move[0]][move[1]].value == 0 or square[move[0]][move[1]].value == self.position:
                    evaluation += 1

        return evaluation


    """
    Return a value corresponding to how defensive agent
    should play given a game status
    """
    def defensive_evaluation(self, state):
        defensive_coef = 1/12  # how safe the agent plays
        if state.phase == 2:
            score = state.get_player_info(self.position)["score"]
            opp_score = state.get_player_info(self.position*-1)["score"]
            balance = score - opp_score

            if SeegaRules.is_end_game(state) and balance < 0:
                return float('-inf')
            elif SeegaRules.is_end_game(state) and balance > 0:
                return float('inf')
            else:
                return defensive_coef + defensive_coef * self.safety_evaluation(state)
        else:
            return 0

    """
    return the number of conner & edges
    """
    def corner_edges(self, state):
        dimension = state.board.board_shape
        corners = [(0, 0), (dimension[0], 0), (0, dimension[1]), (dimension[0], dimension[1])]
        corner, edges = 0, 0

        pieces_on_board = state.board.get_player_pieces_on_board(Color(self.position))
        for piece in pieces_on_board:
            if piece in corners:
                corner += 1
            elif piece[0] == 0 or piece[0] == dimension[0]:
                edges += 1
            elif piece[1] == 0 or piece[1] == dimension[1]:
                edges += 1
        return corner, edges

    """
    return the max number of pieces "connected" for a
    given color
    """
    def get_conn_pieces_num(self, state, color):

        all_pieces = state.board.get_player_pieces_on_board(color)
        connected_pieces = list()
        max_connections = 0

        for piece in all_pieces:
            if piece not in connected_pieces:
                new_connected_pieces = [piece]
                list_piece_connections(piece, all_pieces, new_connected_pieces)
                if len(new_connected_pieces) > max_connections:
                    connected_pieces = new_connected_pieces

        return len(connected_pieces)


    """
    Return number of opponents' possibles and
    maximums captures for given state
    """
    def opponent_captures(self, state):
        dimension = state.board.board_shape
        square = state.board.get_board_state()
        opp_color = self.position*-1
        player_color = self.position
        opp_max_cap = 0
        opp_possible_cap = 0

        opp_pieces_on_board = state.board.get_player_pieces_on_board(Color(self.position*-1))
        for piece in opp_pieces_on_board:
            moves = SeegaRules.get_effective_cell_moves(state, piece)
            if len(moves) > 0:
                move_threat = 0
                for move in moves:
                    # map two steps position
                    gaps = [(move[0] + a[0], move[1] + a[1]) for a in [(0, 2), (0, -2), (2, 0), (-2, 0)]
                            if (0 <= move[0] + a[0] < dimension[0]) and (0 <= move[1] + a[1] < dimension[1])]
                    # map one steps position
                    neig = [(move[0] + a[0], move[1] + a[1]) for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            if ((0 <= move[0] + a[0] < dimension[0]) and (0 <= move[1] + a[1] < dimension[1])
                            and ((move[0] + a[0], move[1] + a[1]) != (dimension[0]//2, dimension[1]//2)))]

                for i in range(len(moves)):
                    if i < len(gaps) and i < len(neig):
                        if square[gaps[i][0]][gaps[i][1]]==(opp_color) and square[neig[i][0]][neig[i][1]]==player_color:
                            move_threat += 1
                            opp_possible_cap += 1

                if move_threat > opp_max_cap:
                    opp_max_cap = move_threat

        return opp_possible_cap, opp_max_cap

    """
    * The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        evaluate_score = 0
        if state.phase==2:

            # Defensive index
            defensive_idx = self.defensive_evaluation(state)

            # Pieces Balance
            age_pieces = len(state.board.get_player_pieces_on_board(Color(self.position)))
            opp_pieces = len(state.board.get_player_pieces_on_board(Color(self.position*-1)))

            # Corner and Edges
            age_conners, age_egdes = self.corner_edges(state)
            opp_corners, opp_edges = self.corner_edges(state)

            # Opponents threats
            ## opp_possible_cap = self.opponent_captures(state)[0]
            opp_max_cap = self.opponent_captures(state)[1]

            # Connected pieces
            age_structure = self.get_conn_pieces_num(state, Color(self.position))
            opp_structure = self.get_conn_pieces_num(state, Color(self.position*-1))

            evaluate_score += (age_pieces - opp_pieces)*5 # Difference in pieces
            evaluate_score += age_conners*2 # Count corner pieces 3 times in total as they are uncapturable
            evaluate_score += age_egdes # Count edge pieces 2 times in total as they are more difficult to capture than center pieces but still capturable
            evaluate_score -= opp_edges # Decrease for opponent edge pieces as if we can capture some it is very good. No decrease for corner pieces as we can not capture them
            evaluate_score -= opp_max_cap # Decrease for direct captures the opponent can make
            evaluate_score += age_structure # Increase for the size of our biggest structure
            evaluate_score -= opp_structure # Decrease for the size of the opponent structure
            evaluate_score += 2 * defensive_idx

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
