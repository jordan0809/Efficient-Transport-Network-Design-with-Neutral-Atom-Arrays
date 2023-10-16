# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:10:10 2023

@author: lai, Chia-Tso
"""

import numpy as np
import pandas as pd
from dimod import Binary
import networkx as nx
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler
from dimod.binary import quicksum
from copy import deepcopy



#Make the adjacent matrix of a graph
def adjacent_matrix(graph,n):

    adjacent = np.zeros((n,n))
    for edge in graph:
        adjacent[edge[0],edge[1]] = 1
        adjacent[edge[1],edge[0]] = 1
    
    return adjacent



#Assign lines by minimizing the difference in length of different lines
def train_line_equal_length(n,nx_graph):  #Wn here is the weight of the station
    
    cqm = ConstrainedQuadraticModel()
    
    A = adjacent_matrix(list(nx_graph.edges),n)
    
    terminal = [i for i in range(n) if np.sum(A[i,:])%2 ==1] #terminal stations
    nt = len(terminal)
    
    #make the distance matrix of terminals
    length = [{k:nx.shortest_path_length(nx_graph,t,k,weight="weight") for k in terminal} for t in terminal]
    d = np.zeros((nt,nt))
    for i in range(nt):
        for j in range(nt):
            d[i][j] = length[i][terminal[j]]
    
    #upper traingle of d
    w = np.array([d[i][j] for i in range(nt) for j in range(i+1,nt)])
    
    x = [0 if (i>j) or (i==j) else Binary(f"x{i}{j}") for i in range(nt) for j in range(nt)]
    x = np.array(x).reshape(nt,nt)
    x = [x[j][i] if i>j else x[i][j] for i in range(nt) for j in range(nt)]
    x = np.array(x).reshape(nt,nt)
    
    upper_x = np.array([x[i,j] for i in range(nt) for j in range(i+1,nt)])
    
    wx = w*upper_x
    
    #minimize the length difference between each line
    term1 = quicksum([(wx[k]-wx[s])**2 for k in range(len(wx)) for s in range(k+1,len(wx))])
    #minus the chosen-not chosen term
    term2 = quicksum([wx[k]**2 for k in range(len(wx))])*nt*(0.5*nt-1)   #nt*(0.5*nt-1) is the num of non-picked pair
    #minimize the overhead in connections as well
    term3 = quicksum(wx)
    cqm.set_objective(term1-term2+term3)
    
    #each terminal can only be picked once
    for i in range(nt):
        cqm.add_constraint(quicksum([x[i,j] for j in range(nt)]) == 1)
        
    
    token = "Insert Dwave API token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="line_assignment")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    data = pd.DataFrame(feasible_sampleset)
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_dict = dict(data.iloc[best_index,:])
    
    lines = []
    for i in range(nt):
        for j in range(i+1,nt):
            if best_dict[f"x{i}{j}"] == 1:
                lines.append(nx.shortest_path(nx_graph,terminal[i],terminal[j],weight="weight"))
    
    # connections between odd degree(except for 1) nodes might be redundant lines
    refined = deepcopy(lines)
    odd = [i for i in range(n) if (np.sum(A[i,:])%2 ==1) and (np.sum(A[i,:])!=1) ]
    for line in refined:
        if (line[0] in odd) and (line[-1] in odd):
            refined.remove(line)
    
    refinedset = [set(line) for line in refined]
    k = set().union(*refinedset)
    if k.issubset(set(range(n))):
        lines = refined
    
    return lines