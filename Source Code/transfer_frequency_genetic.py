# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:03:55 2023

@author: Lai,Chia-Tso
"""


import numpy as np
import networkx as nx
import random
from copy import deepcopy



def adjacent_matrix(graph,n):

    adjacent = np.zeros((n,n))
    for edge in graph:
        adjacent[edge[0],edge[1]] = 1
        adjacent[edge[1],edge[0]] = 1
    
    return adjacent


def generate_population(n,nx_graph,popu_size):
    #create a list of odd connection nodes
    terminal = [i for i in range(n) if np.sum(adjacent_matrix(list(nx_graph.edges),n)[i,:])%2 ==1]
    # shuffle the nodes randomly to form different lines
    population = [random.sample(terminal,len(terminal)) for i in range(popu_size)]
    return population



#transfer time from a source to all the nodes in the graph
def single_source_transfer_freq(nx_graph,chromo,start):
    
    n = len(nx_graph.nodes)
    lines = [set(nx.shortest_path(nx_graph,chromo[2*i],chromo[2*i+1])) for i in range(int(len(chromo)/2))]
    nl = len(lines)

    stations = []
    for i in range(nl):
        stations += list(lines[i])
         
    if set(stations) != set(range(n)): #if the lines do not cover all the stations then it's invalid
        return  "not every station is covered"
    
    else:
        
    #adjacet matrix of lines
        ad = np.zeros((nl,nl))
        for i in range(nl):
            for j in range(nl):
                if len(lines[i].intersection(lines[j])) != 0:
                    ad[i,j] = 1
                else:
                    ad[i,j] = 0
        for i in range(nl):
            ad[i,i] = 0

        start_line = set([i for i in range(nl) if start in lines[i]]) #lines that go through start station
    
        reachable = set().union(*[lines[i] for i in list(start_line)]) #reachable stations from the start
        rounds = [reachable]
        current = [i for i in list(start_line)]  #current lines
        explored_lines = list(start_line)
    
        while set(range(n)).issubset(reachable) == False:
            neighbors = [set(np.where(ad[i,:] == 1)[0]) for i in current]
            neighbors = list(set().union(*neighbors)-set(explored_lines))
            reachable = reachable.union(*[lines[k] for k in neighbors])
            rounds.append(reachable)
            current = neighbors
            explored_lines += neighbors
            if reachable == set(range(n)):
                break
            
        transfers = [rounds[0] if i==0 else rounds[i]-rounds[i-1] for i in range(len(rounds))]
    
        d = np.zeros(n)
        for index,item in enumerate(transfers):
            for station in list(item):
                d[station] = index
        return d


#Average transfer times for any pair of nodes in the graph
def avg_transfer_freq(nx_graph,chromo):
    
    n = len(nx_graph.nodes)
    
    lines = [set(nx.shortest_path(nx_graph,chromo[2*i],chromo[2*i+1])) for i in range(int(len(chromo)/2))]
    nl = len(lines)
    
    stations = []
    for i in range(nl):
        stations += list(lines[i])
         
    if set(stations) != set(range(n)): #if the lines do not cover all the stations then it's invalid
        return  100000
    
    else:
        
        distance = np.array([list(single_source_transfer_freq(nx_graph,chromo,start)) for start in range(n)])
    
        summ = np.sum(distance[np.triu_indices(n)])
        norm = n*(n-1)/2
        avg_freq = summ/norm
    
        return avg_freq



#evaluate the transfer time and the increase in overall length
def fitness(nx_graph,W,chromo):
    
    n = len(nx_graph.nodes)
    
    lines = [nx.shortest_path(nx_graph,chromo[2*i],chromo[2*i+1]) for i in range(int(len(chromo)/2))]
    nl = len(lines)
    
    stations = []
    for i in range(nl):
        stations += lines[i]
         
    if set(stations) != set(range(n)): #if the lines do not cover all the stations then it's invalid
        fitness = 10000000
        
    else:
        fitness = avg_transfer_freq(nx_graph,chromo)
    
    #calculate increased overall length of the network
    lines_length = [nx.shortest_path_length(nx_graph,chromo[2*i],chromo[2*i+1],weight="weight") for i in range(int(len(chromo)/2))]
    plus = np.sum(lines_length) - np.sum([W[u][v] for u,v in list(nx_graph.edges)])
    
    return fitness+plus/10
    


def selection(popu,fitness,nx_graph,W,selection_rate):
    sample_size = round(len(popu)*selection_rate)
    fitness_score = [fitness(nx_graph,W,chromo) for chromo in popu]
    best_chromo = np.argsort(fitness_score)[:sample_size]  #take out the smaller ones
    parents = [popu[i] for i in best_chromo]
    return parents


def crossover(parents,popu_size):
    
    offspring = parents
    for i in range(popu_size-len(parents)):
        par = random.sample(range(len(parents)),1)[0]
        selected = parents[par]
        indices = random.sample(range(len(parents[0])),2)
        cross_par = deepcopy(selected)
        cross_par[indices[0]],cross_par[indices[1]] = cross_par[indices[1]],cross_par[indices[0]]
        offspring.append(cross_par)
    return offspring


#the standard deviation of the top 30% of the population
def top_std(popu,fitness,nx_graph,W):
    fitness_score = [fitness(nx_graph,W,chromo) for chromo in popu]
    return np.std(np.sort(fitness_score)[:round(len(popu)*0.3)])


def train_line_genetic(n,nx_graph,popu_size,fitness,W,selection_rate,max_iteration,cutoff_std):
    
    popu = generate_population(n,nx_graph,popu_size)
    
    best_individual=[]
    mini_fit = []
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        parents = selection(popu,fitness,nx_graph,W,selection_rate)
        offspring = crossover(parents,popu_size)
        popu = offspring
        
        fitness_score = [fitness(nx_graph,W,chromo) for chromo in popu]
        best = popu[np.argmin(fitness_score)]
        mini_fit.append(np.min(fitness_score))
        best_individual.append(best)
        
        if top_std(popu,fitness,nx_graph,W) < cutoff_std:
            break
    
    optimal = best_individual[np.argmin(mini_fit)]
    optimal = [nx.shortest_path(nx_graph,optimal[2*i],optimal[2*i+1]) for i in range(int(len(optimal)/2))]
    print("minimum cost:",np.min(mini_fit))
    print("optimal lines:",optimal)
    print("iteration:",iteration)
    
    return [optimal,mini_fit]



graph100 = [(12, 68), (12, 69), (12, 70), (16, 60), (19, 38), (25, 31), (25, 38), (31, 88), (35, 69), (38, 57), (38, 91), (38, 62), (42, 67), (42, 80), (42, 91), (45, 60), (50, 72), (57, 60), (57, 68), (60, 72), (70, 86), (72, 83), (72, 66), (83, 90), (84, 85), (85, 88), (88, 29), (6, 48), (6, 92), (9, 37), (9, 41), (11, 92), (29, 59), (29, 64), (29, 93), (29, 95), (29, 99), (29, 81), (29, 73), (30, 77), (30, 87), (34, 53), (34, 77), (40, 64), (41, 48), (41, 97), (41, 99), (43, 92), (44, 63), (44, 64), (46, 63), (52, 87), (52, 93), (1, 3), (1, 5), (1, 62), (1, 96), (2, 14), (2, 15), (2, 81), (3, 28), (3, 62), (4, 96), (5, 7), (13, 15), (13, 33), (14, 65), (15, 17), (15, 49), (22, 26), (22, 49), (22, 62), (24, 98), (36, 55), (47, 81), (49, 20), (55, 96), (62, 74), (96, 98), (0, 61), (8, 39), (8, 76), (8, 79), (8, 94), (10, 58), (18, 58), (20, 21), (20, 56), (20, 66), (20, 76), (21, 58), (21, 75), (21, 78), (23, 54), (27, 76), (27, 89), (32, 73), (51, 89), (54, 71), (58, 61), (71, 73), (71, 82), (73, 76)]
W100 = np.random.randint(1,100,size=(100,100))
g100 = nx.Graph()
for i in range(100):
    g100.add_node(i)
for j in graph100:
    g100.add_edge(j[0],j[1],weight=W100[j[0]][j[1]])
    

graph30 = [(0, 27), (1, 21), (1, 29), (2, 5), (3, 5), (3, 19), (4, 16), (4, 24), (5, 12), (5, 15), (6, 8), (7, 22), (7, 24), (8, 17), (9, 19), (9, 28), (10, 13), (11, 16), (13, 24), (13, 27), (14, 19), (17, 22), (17, 26), (17, 28), (18, 24), (19, 25), (20, 28), (23, 26), (27, 29)]
W30 = np.random.randint(1,30,size=(30,30))
g30 = nx.Graph()
for i in range(30):
    g30.add_node(i)
for j in graph30:
    g30.add_edge(j[0],j[1],weight=W30[j[0],j[1]])


popu_size = 1000
experiment = train_line_genetic(100,g30,popu_size,fitness,W100,selection_rate=0.3,max_iteration=100,cutoff_std=0.1)
print(experiment[0])
print(experiment[1])