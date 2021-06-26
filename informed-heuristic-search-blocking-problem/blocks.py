# -*-coding: utf-8 -*
import sys
from search import *
import time
import copy
from collections import Counter


#################
# Problem class #
#################
class Blocks(Problem):

    def successor(self, state):
        print("Node")
        print(state.__str__())
        successors = self.get_successors(state)
        # print('Number of successors:', len(successors))

        for i, successor in enumerate(successors):

            # print('child:', i+1)
            # print(State(successor))
            yield (successor, State(successor))

    def goal_test(self, state):
        'verify is all goal_grid letters are @ in current_grid'
        count = 0

        for l in state.grid:
            for i in l:
                if i == '@':
                    count += 1

        letters, unmovable = self.get_grid_positions(self.goal.grid)


        return count == len(letters)

    def path_cost(self, c, state1, action, state2):
        #return c + abs(state1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        return c + 1


    ###########################
    # Blocks uxiliars methods
    ###########################
    # aux method 1
    def get_successors(self, state):

        cur_letters, unmovable = self.get_grid_positions(state.grid) # aux method 2 [[a,2,4,], [a,4,4,], [b,3,4]]
        upd_grids = self.get_moves(state, unmovable, cur_letters) # aux method 3

        return upd_grids

    # aux method 2
    def get_grid_positions(self, grid):
        # list movings blocks:
        unmovable = [' ', '#', '@']
        # get letters current position
        letters = []
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] not in unmovable:
                    letters.append([grid[row][col], row, col])
        return letters, unmovable

    # aux method 3
    def get_moves(self, state,  unmovable, letters):
        n_grids = []
        grid = state.grid
        goal_grid = (self.goal).grid
        moves = [(0, 1), (0, -1)]
        for letter, row, col in letters: #[[a,2,4,], [a,4,4,], [b,3,4]]
            for i,j in moves:
                temp_grid = copy.deepcopy(grid)
                if (0 <= col+j < state.nbc) and (grid[row+i][col+j] == ' '):
                    'if not in the space bottom and if underneath is empty '
                    temp_grid[row][col] = ' '
                    while (row + i + 1< state.nbr) and (grid[row + i + 1][col+j] == ' '):
                        i += 1
                    if (row+i < state.nbr and row+i >= 0) and goal_grid[row + i][col+j]  == letter.upper():
                        temp_grid[row+i][col+j] = '@'
                        temp_grid = self.get_gravity(temp_grid)
                    elif (row+i < state.nbr and row+i >= 0):
                        temp_grid[row + i][col+j] = letter
                        temp_grid = self.get_gravity(temp_grid)
                    if self.dead_lock(goal_grid, letter, row + i , col+j, temp_grid):  # aux method 4
                        next
                    else:
                        n_grids.append(temp_grid)
        return n_grids

    # aux method 4
    def dead_lock(self, goal_grid, cur_letter, cur_row, cur_col, temp_grid):

        target_letters, unmovable = self.get_grid_positions(goal_grid) # aux method 2
        current_letters, unmovable = self.get_grid_positions(temp_grid)

        n_target = Counter(target_letter[0] for target_letter in target_letters) # dict
        n_letter = Counter(current_letter[0] for current_letter in current_letters) # dict

        n = len(target_letters)
        m = len(current_letters)

        for i in range(n):
            for j in range(m):
                target, t_row, t_col = target_letters[i] # [[A,5,4,],[B,3,4]]
                current, c_row, c_col = current_letters[j] # [[a,2,4,], [a,4,4,], [b,3,4]]
                if current.upper() == target:
                    if c_row > t_row:  # current below do target
                        n_letter[current] -= 1
                    elif c_row == t_row: # same level as target, but different column
                        if c_col >= t_col:
                            l = t_col
                            r = c_col
                        else:
                            r = t_col
                            l = c_col
                        for x in range(l+1, r):
                            if goal_grid[c_row][x] != " ":
                                n_letter[current] -= 1

        for k in n_target.keys():
            if n_letter[k.lower()] < 0:
                # print('Deadlock')
                # print(State(temp_grid))
                return True


    # aux method 5
    def get_gravity(self, grid):
        unmovable = [' ', '#', '@']
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                i = 1
                if grid[row][col] not in unmovable and row + i < len(grid) and grid[row + i][col] == ' ' :
                    while row + i + 1 < len(grid)  and grid[row + i + 1][col] == ' ':
                        i += 1
                    grid[row + i][col] = str(grid[row][col])
                    grid[row][col] = ' '
        return grid

    # aux method 4
    def heuristic(self, node):
        h = 0.0
        grid = node.state.grid
        goal_grid = (self.goal).grid

        current, unmovable = self.get_grid_positions(grid) # aux method 2

        for row in range(len(goal_grid)):
            for col in range(len(goal_grid[0])):
                if grid[row][col] != '@' and goal_grid[row][col] != '#' and goal_grid[row][col] != ' ':
                    h = h + 15
                    penalty = 1e5 #penalize cases when target is above the current position
                    for letter, c_row, c_col in current:
                        if grid[c_row][c_col].upper() == goal_grid[row][col] and c_row <= row:
                            penalty = abs(row - c_row,) + abs(col - c_col)
                    h = h + penalty
        return h

###############
# State class #
###############
class State:
    def __init__(self, grid):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid

    def __str__(self):
        n_sharp = self.nbc + 2
        s = ("#" * n_sharp) + "\n"
        for i in range(self.nbr):
            s += "#"
            for j in range(self.nbc):
                s = s + str(self.grid[i][j])
            s += "#"
            if i < self.nbr - 1:
                s += '\n'
        return s + "\n" + "#" * n_sharp


    ###########################
    # States uxiliars methods
    ###########################
    # aux method 4
    def add_goal(self, grid):
        # embedding goal into state
        self.goal = grid


    def __eq__(self, state2):
        return self.grid == state2.grid


    def __hash__(self):
        return hash(self.__str__())

######################
# Auxiliary function #
######################
def readInstanceFile(filename):
    grid_init, grid_goal = map(lambda x: [[c for c in l.rstrip('\n')[1:-1]] for l in open(filename + x)], [".init", ".goalinfo"])
    return grid_init[1:-1], grid_goal[1:-1]

######################
# Heuristic function #
######################
def heuristic(node):

    grid = node.state.grid
    goal_grid = (goal_state).grid
    # fix current
    current = get_blocks(grid) # list if block position ['block', x,y] !!! use strcuture to compare letter current and target


    h = 0
    for row in range(len(goal_grid)):
        for col in range(len(goal_grid[0])):
            penalty = 0
            # check if blocks in the goal state are @ in the current
            if grid[row][col] != '@' and (goal_grid[row][col] != '#' and goal_grid[row][col] != ' '):
                h = h + 15

                count_above =0
                for letter, c_row, c_col in current:
                    # number of target for a given letter B
                    # how many box for a given letter not lower than the target!!!

                    # check if the destination is for the letter

                    if goal_grid[row][col] == grid[c_row][c_col].upper():
                        # at least one same letter on top
                        # if the current letter position is above the target letter position
                        if c_row <= row:
                            count_above +=1

                if count_above >= 1:
                    penalty = penalty + abs(row - c_row,) + abs(col - c_col)
                else:
                    penalty = 1e5

            h = h + penalty

    print(node.state.__str__())
    print('Evaluations',h)
    print(goal_state)
    return h

def get_blocks(grid):
    # list movings blocks:
    unmovable = [' ', '#', '@']
    # get letters current position
    letters = []
    for idRow, row in enumerate(grid):
        for idCol,value in enumerate(row):
            if value not in unmovable:
                col = row.index(value)
                letters.append([value, idRow, idCol])
    return letters



##############################
# Launch the search in local #
##############################
#Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious
instances_path = "/mnt/c/Users/b_tib/coding/Msc/oLINGI2261/Assignements/artificial-intelligence/informed-heuristic-search-Blocking-Problem/instances/"

instance_names = [ 'a06', 'a07','a08','a09','a10','a11'] #

for instance in [instances_path + name for name in instance_names]:
    grid_init, grid_goal = readInstanceFile(instance)

    init_state = State(grid_init)
    goal_state = State(grid_goal)

    problem = Blocks(init_state, goal_state)

    # print(init_state)
    # print(goal_state)

    # example of bfs tree search
    # startTime = time.perf_counter()
    # node, nb_explored, remaining_nodes = breadth_first_graph_search(problem)
    # endTime = time.perf_counter()
    # example of astar graph search
    startTime = time.perf_counter()
    node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
    endTime = time.perf_counter()

    # example of print
    path = node.path()
    path.reverse()



    print('Number of moves: ' + str(node.depth))
    for n in path:
        print(n.state)  # assuming that the __str__ function of state outputs the correct format
        print()
    print("* Execution time:\t", str(endTime - startTime))
    print("* Path cost to goal:\t", node.depth, "moves")
    print("* #Nodes explored:\t", nb_explored)
    print("* Queue size at goal:\t",  remaining_nodes)



####################################
# Launch the search for INGInious  #
####################################
# #Use this block to test your code on INGInious
# instance = sys.argv[1]
# grid_init, grid_goal = readInstanceFile(instance)
# init_state = State(grid_init)
# goal_state = State(grid_goal)
# problem = Blocks(init_state)

# # example of bfs graph search
# startTime = time.perf_counter()
# node, nb_explored, remaining_nodes = astar_graph_search(problem, heuristic)
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
