#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Gaël Aglin <gael.aglin@uclouvain.be>"""
from search import *
import sys
import random
import numpy as np
from copy import deepcopy
class BinPacking(Problem): 

    def successor(self, state): # feasible states 
        def get_bin(item_key): 
            '''returns the indice of the bin in which an item is'''
            index = 0 
            for bin in state.bins : 
                if item_key in bin: 
                    return index, bin
                index += 1 

        # Swap 2 items---------------------------------------------------------
        for (key1,val1) in list(state.items.items()) : 
            for (key2,val2) in list(state.items.items()) : 
                #items should be different and the pair should appear once, in different bins 
                id1, bin1 = get_bin(key1)
                id2, bin2 = get_bin(key2)
                if  id1 != id2 and state.can_fit(bin1, val2-val1) and state.can_fit(bin2, val1-val2):
                    
                    updated = deepcopy(state.bins)
                    del updated[id1][key1]
                    updated[id1][key2] = val2
                    del updated[id2][key2]
                    updated[id2][key1] = val1
                    
                    yield 'swap2', State(state.capacity, state.items, updated)
        
            
        
        # Swap 1 item with an empty space -------------------------------------
        for (key,val) in list(state.items.items()):
            id, bin = get_bin(key)
            for b in range(len(state.bins)): 
                if id != b and state.can_fit(state.bins[b],val): 
                    
                    updated = deepcopy(state.bins)
                    del updated[id][key]
                    updated[b][key] = val
                    
                    try :
                        updated.remove({})
                    except ValueError:
                        pass
                    
                    yield 'swap1', State(state.capacity, state.items, updated)
                    
                
    def fitness(self, state): # implementation of the cost function to be min 
        """
        :param state:
        :return: fitness value of the state in parameter
        """
        fit = 0
        for bin in state.bins : 
            full = 0 
            for val in bin.values():
                full += val
            fit += (full/state.capacity)**2 #use fullness ! 
        fit = 1 - fit/len(state.bins)
        return -fit

class State:

    def __init__(self, capacity, items, bins=None):
        self.capacity = capacity
        self.items = items
        if bins == None:
            self.bins = self.build_init()
        else : 
            self.bins = bins 

    # an init state building is provided here but you can change it at will
    def build_init(self):
        init = []
        for ind, size in self.items.items():
            if len(init) == 0 or not self.can_fit(init[-1], size):
                init.append({ind: size})
            else:
                if self.can_fit(init[-1], size):
                    init[-1][ind] = size
        return init

    def can_fit(self, bin, itemsize):
        return sum(list(bin.values())) + itemsize <= self.capacity

    def __str__(self):
        s = ''
        for i in range(len(self.bins)):
            s += ' '.join(list(self.bins[i].keys())) + '\n'
        return s


def read_instance(instanceFile):
    file = open(instanceFile)
    capacitiy = int(file.readline().split(' ')[-1])
    items = {}
    line = file.readline()
    while line:
        items[line.split(' ')[0]] = int(line.split(' ')[1])
        line = file.readline()
    return capacitiy, items

# Attention : The goal is to maximize it --------------------------------------
def maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    fitness = problem.fitness(current.state)
    for step in range(limit):
        if callback is not None:
            callback(current)
        reachable = list(current.expand())
        values = list(map(lambda x: problem.fitness(x.state), reachable))
        candidate = np.argmax(values)

        current = reachable[candidate]
        
        if values[candidate] > fitness:  #fitness of best candidate 
            best = LSNode(problem, current.state,step + 1)
            fitness = values[candidate]
        
    return best
# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it


# Attention : The goal is to minimize it --------------------------------------
def randomized_maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    fitness = problem.fitness(current.state)
    for step in range(limit):
        if callback is not None:
            callback(current)
        reachable = list(current.expand())
        values = list(map(lambda x: problem.fitness(x.state), reachable))
        candidates = np.argpartition(values, -5)[-5:]
        candidate = np.random.choice(candidates)
        current = reachable[candidate]

        if values[candidate] > fitness:  #fitness of best candidate 
            best = LSNode(problem, current.state,step + 1)
            fitness = values[candidate]

    return best


#####################
#       Launch      #
#####################
if __name__ == "__main__":
    info = read_instance(sys.argv[1])
    init_state = State(info[0], info[1])
    bp_problem = BinPacking(init_state)
    step_limit = 100
    node = maxvalue(bp_problem, step_limit)
    state = node.state
    print(state)
