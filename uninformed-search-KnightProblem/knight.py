# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin <gael.aglin@uclouvain.be>'''
import time
import sys
from search import *
import pandas as pd

#################
# Problem class #
#################
class Knight(Problem):
    '''
    Notations:
    px, py: previous or parent board position
    x, y: previous or parent board position
    sx, sy: successor board position
    '''

    def successor(self, state):
        '''
        state: (current) node state within the problem
        seq: 0, 1, 2, 3
        '''
        seq = 4

        (x, y) = (state.curr_x, state.curr_y)
        sucessors = self.feasible_successors(x, y, state, seq)

        # Warnsdorff's rule: sort sucessor by decreasing number of sucessors sucessors
        if seq == 4:
            sucessors.sort(key=lambda t: -len(self.feasible_successors(t[0], t[1], state, seq)))
        for (sx, sy) in sucessors:
            # list of ()
            yield ((sx, sy), self.sucessor_state(x, y, sx, sy, state))


    def feasible_successors(self, x, y, state, seq):
        '''
        Will tried here different sequences
        '''
        # Listofmov A B C D
        ABCD = ((-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
        # Listofmov B A C D
        BACD = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
        # Listofmov B C A D
        BCAD =((1, 2), (1, -2),  (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1))
        # Clockwise
        clock_wise = ((2, 1), (1, 2),  (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))

        sequences = {0: ABCD, 1: BACD, 2: BCAD, 3: clock_wise, 4: clock_wise}

        return [(x+i, y+j) for (i,j) in sequences[seq] if 0 <= x+i < state.nRows and 0 <= y+j < state.nCols and state.grid[x+i][y+j] == " "]

    def sucessor_state(self, x, y, sx, sy, state):
        '''
        '''
        next_state = State([state.nRows, state.nCols], (sx, sy), False)
        next_state.grid = [x[:] for x in state.grid]
        next_state.grid[x][y] = u"\u265E"
        next_state.grid[sx][sy] = u"\u2658"

        return next_state

    def goal_test(self, state):
        """
        Check the chess board is complete -> No " " in the state.grid()
        state: (current) node state within the problem
        """
        for row in state.grid:
            if " " in row:
                return False
        return True

###############
# State class #
###############

class State:
    '''
    curr_x: current x coordenate
    curr_y: current y coordenate
    '''
    def __init__(self, shape, init_pos, new_board=True):
        self.nRows = shape[0]
        self.nCols = shape[1]
        self.curr_x = init_pos[0]
        self.curr_y = init_pos[1]
        self.grid = []
        if new_board:
            for i in range(self.nRows):
                self.grid.append([" "]*self.nCols)
            self.grid[init_pos[0]][init_pos[1]] = "♘"

    def __str__(self):
        n_sharp = 2 * self.nCols + 1
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nRows):
            s += "#"
            for j in range(self.nCols):
                s = s + str(self.grid[i][j]) + " "
            s = s[:-1]
            s += "#"
            if i < self.nRows - 1:
                s += '\n'
        s += "\n"
        s += "#" * n_sharp
        return s


    def __hash__(self):
        '''
        What make a state equal to another
        '''
        return hash((self.nRows, self.nCols, tuple(map(tuple, self.grid))))

    def __eq__(self, other):
        '''
        What make a state equal to another
        '''
        return (self.nRows, self.nCols, self.grid) == (other.nRows, other.nCols, other.grid)


##############################
# Launch the search in local
##############################
#Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious
with open('/mnt/c/Users/b_tib/coding/Msc/oLINGI2261/Assignement1/instances.txt') as f:
    instances = f.read().splitlines()

    searches = {'DFT': depth_first_tree_search
            , 'BFT': breadth_first_tree_search
            , 'BFG': breadth_first_graph_search
            , 'DFG': depth_first_graph_search
        }

    frame = pd.DataFrame(columns = ["Search", "Instance", "T",  "EN", "RNQ", "Moves"])

    i = 0
    for search, f_search in searches.items():

        for instance in instances:
            elts = instance.split(" ")
            shape = (int(elts[0]), int(elts[1]))
            init_pos = (int(elts[2]), int(elts[3]))
            init_state = State(shape, init_pos)

            problem = Knight(init_state)

            startTime = time.perf_counter()
            node, nb_explored, remaining_nodes = f_search(problem)
            endTime = time.perf_counter()

            if node:
                moves = node.depth
                path = node.path()
                path.reverse()
            else:
                moves = 'no solution'

            frame.loc[i] = [search, i%10, endTime - startTime, nb_explored, remaining_nodes, moves]
            i += 1


            # for n in path:
            #     print(n.state)  # assuming that the __str__ function of state outputs the correct format
            #     print()

            print('Search: ' + str(search))
            print('Instance: ' + str(elts))
            print("* Execution time:\t", str(endTime - startTime))
            print("* Path cost to goal:\t", moves, "moves")
            print("* #Nodes explored:\t", nb_explored)
            print("* Queue size at goal:\t",  remaining_nodes)
    print(frame)

####################################
# Launch the search for INGInious  #
####################################
# Use this block to test your code on INGInious
# shape = (int(sys.argv[1]),int(sys.argv[2]))
# init_pos = (int(sys.argv[3]),int(sys.argv[4]))
# init_state = State(shape, init_pos)

# problem = Knight(init_state)

# # example of bfs tree search
# startTime = time.perf_counter()
# node, nb_explored, remaining_nodes = depth_first_graph_search(problem)
# endTime = time.perf_counter()

# # example of print
# path = node.path()
# path.reverse()

# print('Number of moves: ' + str(node.depth))
# for n in path:
#     print(n.state)  # assuming that the __str__ function of state outputs the correct format
#     print()
# print("* Execution time:\t", str(endTime - startTime))
# print("* Path cost to goal:\t", node.depth, "moves")
# print("* #Nodes explored:\t", nb_explored)
# print("* Queue size at goal:\t",  remaining_nodes)
