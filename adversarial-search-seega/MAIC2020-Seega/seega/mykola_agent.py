from core.player import Player, Color
from seega.seega_rules import SeegaRules
from copy import deepcopy
from time import time


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class AI(Player):

    in_hand = 12
    score = 0
    name = "V5"
    max_depth = 1

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value
        self.flat_max_depth = 0

    def play(self, state, remain_time):
        min_moves_vict = 10 * (state.MAX_SCORE - state.get_player_info(self.position)["score"])
        if state.phase > 1:
            while True:
                t0 = time()
                # print(f"search at depth {self.max_depth}")
                res = minimax_search(state, self)
                dt = time() - t0
                remain_time -= dt
                if min_moves_vict * dt > remain_time:
                    break
                else:
                    self.max_depth += 1
        else:
            res = minimax_search(state, self)
        
        self.max_depth = 1
        return res

    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        children = []
        player = state.get_next_player()
        for action in SeegaRules.get_player_actions(state, player):
            new_state = deepcopy(state)
            if SeegaRules.act(new_state, action, player):
                children.append((action, new_state))
        
        children.sort(key=lambda x: self.evaluate(x[1]), reverse=player != self.position)
        return children

    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        return SeegaRules.is_end_game(state) or depth >= self.max_depth
    
    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        score = state.get_player_info(self.position)["score"]
        other_score = state.get_player_info(-self.position)["score"]
        
        if state.phase == 1:
            # placement phase heuristic
            return 0
        elif SeegaRules.is_end_game(state):
            return 0xFFFFFFFF * (score - other_score)
        else:
            # normal capture phase heuristic
            return (score - other_score) + 0.5 * self.get_safe_count(state.board)


    def get_safe_count(self, board):
        bs = board.get_board_state()
        score = 0
        increment = 0.25
        safe = (self.position, 0)
        for p in board.get_player_pieces_on_board(Color(self.position)):
            x = p[1]
            y = p[0]
            # TODO also add increment to score if we're at a border/corner because it is safe
            if (y + 1 < board.board_shape[1] and bs[x][y + 1].value in safe):
                score += increment
            if (y - 1 < board.board_shape[1] and bs[x][y - 1].value in safe):
                score += increment
            if (x + 1 < board.board_shape[0] and bs[x + 1][y].value in safe):
                score += increment
            if (x - 1 < board.board_shape[0] and bs[x - 1][y].value in safe):
                score += increment
            
            # # also add increment to score if we're at a border/corner because it is safe
            # if x == 0:
            #     score += increment
            # if x == board.board_shape[0] - 1:
            #     score += increment
            # if y == 0:
            #     score += increment
            # if y == board.board_shape[1] - 1:
            #     score += increment

        return score


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