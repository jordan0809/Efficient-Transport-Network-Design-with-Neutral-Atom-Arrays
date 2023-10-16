# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:57:33 2023

@author: Lai, Chia-Tso
"""

import numpy as np
import pandas as pd
from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler
from dimod.binary import quicksum
from itertools import groupby


#Return connected graphs from a list of graphs
def connected_graph(n,graph_list):
    
    valid_graph = []
    for graph in graph_list:
    
        adjacent = np.zeros((n,n))
        for edge in graph:
            adjacent[edge[0],edge[1]] = 1
            adjacent[edge[1],edge[0]] = 1
        
        power = [np.linalg.matrix_power(adjacent, k) for k in range(1,n)]
        summ = 0
        for p in power:
            summ += p
        validity = np.prod(summ)
        if validity != 0:
            valid_graph.append(graph)
    return valid_graph



#Return a connected network with minimum overall distance while satisfying the constraints
#Biggest stations are required to have at least 4 connections while smallest ones can have at most 2
def mini_overall_dist(n,We,Wn):
    
    cqm = ConstrainedQuadraticModel()
    
    largest = np.argsort(Wn)[-1:-1-round(0.1*n):-1]
    smallest = np.argsort(Wn)[:round(0.1*n)]
    
    x=[0 if (i>j) or (i==j) else Binary(f"x{i}{j}") for i in range(n) for j in range(n)]
    x = np.array(x).reshape(n,n)
    x = [x[j][i] if i>j else x[i][j] for i in range(n) for j in range(n)]
    x = np.array(x).reshape(n,n)
    
    dist = quicksum(We[np.triu_indices(n)]*x[np.triu_indices(n)])
    cqm.set_objective(dist)
    
    #Each station needs to be connected at least once
    for i in range(n):
        cqm.add_constraint(quicksum([x[i,j] for j in range(n)]) >= 1)
    # For any connected pair, at least one node of the pair needs to connect with another node
    for i in range(n):
        for j in range(i+1,n):
            cqm.add_constraint(x[i,j]-quicksum([x[i,s]+x[j,k] for s in range(n) for k in range(n) if (s!=j) and (k!=i)])<=0)
    # Total edges needs to be greater or equal to n-1
    #cqm.add_constraint(0.5*quicksum([quicksum([x[i,j] for j in range(n)]) for i in range(n)]) >= n-1)
    cqm.add_constraint(quicksum(x[np.triu_indices(n)]) >= n-1)
    
    #Top 10% largest station needs at least 4 connections(edges) (two lines)
    for i in largest:
        cqm.add_constraint(quicksum([x[i,j] for j in range(n)]) >= 4)
            
    #Bottom 10% smallest station cannot have more than 2 connections
    for i in smallest:
        cqm.add_constraint(quicksum([x[i,j] for j in range(n)]) <= 2)
        
    token = "Insert D Wave API Token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="Metro_Planning")
    
    data = pd.DataFrame([sampleset.record[i][0] for i in range(len(sampleset.record))],columns=sampleset.variables)
    
    #Take out samples that fulfill the constraints and are connected graph
    feasible_index = np.where(sampleset.record.is_feasible == True)[0]
    
    
    optimal_sol = data.iloc[feasible_index,:]
    
    
    #Convert the ouput variables into the correct order
    value_dict = [dict(optimal_sol.iloc[i,:]) for i in range(optimal_sol.shape[0])]
    graph_list = [[(i,j) for i in range(n) for j in range(i+1,n) if value[f"x{i}{j}"]==1] for value in value_dict]
    
    
    graph_list = [k for k,v in groupby(sorted(graph_list))]  #remove repeated graphs
    
    #extract graphs that satisfy the connectivity criteria
    connected_graph_list = connected_graph(n,graph_list)
    
    return connected_graph_list



#calculate the overall distance of a network
def overall_length(graph_list,W):
    return np.array([np.sum([W[edge[0],edge[1]] for edge in graph]) for graph in graph_list])