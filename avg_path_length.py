# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:05:59 2023

@author: Lai, Chia-Tso
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



#Shortest path algorithm with quantum annealing
def shortest_path(n,We,graph,start,end):
    
    adjacent = adjacent_matrix(graph,n)
    
    if adjacent[start,end] == 1:
        path = [start,end]
        if start > end:
            shortest_dist = We[end,start]
        else:
            shortest_dist = We[start,end]
    else:
        
        cqm = ConstrainedQuadraticModel()
    
        valid_weight = We*adjacent
        for i in range(n):
            for j in range(i):
                valid_weight[i,j] = valid_weight[j,i]
            
        y = np.array([Binary(f"y{i}") if (i!=end) and (i!=start) else 1 for i in range(n)])
    
        path_length = 0.5*np.dot(y,np.dot(valid_weight,y.transpose()))
    
        cqm.set_objective(path_length)
    
        term1 = 0.5*np.dot(y,np.dot(adjacent,y.transpose()))
        term2 = quicksum([y[k] for k in range(n) if (k!=end) and (k!=start)])
        cqm.add_constraint(term1-term2 == 1)
        
        #start and end station need to be connected exactly once by other stations on the path
        cqm.add_constraint(quicksum([adjacent[start,k]*y[k] for k in range(n)]) == 1)
        cqm.add_constraint(quicksum([adjacent[end,k]*y[k] for k in range(n)]) == 1)
    
        token = "Insert Dwave API token"
        cqm_sampler = LeapHybridCQMSampler(token=token)
        sampleset = cqm_sampler.sample_cqm(cqm,label="shortest_path")
    
        data = pd.DataFrame([sampleset.record[i][0] for i in range(len(sampleset.record))],columns=sampleset.variables)
    
        #Take out samples that fulfill the constraints and are connected graph
        feasible_index = np.where(sampleset.record.is_feasible == True)[0]
        shortest_dist = np.min(sampleset.record[feasible_index].energy)
        optimal_index = np.where(sampleset.record[feasible_index].energy == np.min(sampleset.record[feasible_index].energy))[0]
        optimal_sol = data.iloc[feasible_index[optimal_index][0],:]
    
    
        #Convert the ouput variables into the correct order
        value_dict = dict(optimal_sol) 
        station = [start,end]
        for i in range(n):
            if (i!=end) and (i!=start):
                if (value_dict[f"y{i}"] == 1):
                    station.append(i)
        
        #reorder the stations to form a path
        path = [start]
        num = len(station)

        while len(path) < num:
            for s in station[1:]:
                if (adjacent[s,path[-1]] == 1) and (s not in path):
                    path.append(s)
        
    
        shortest_dist = np.sum([We[path[i],path[i+1]] for i in range(len(path)-1)])
    
    return [path,shortest_dist]


#Single source shortest path algorithm
def ss_shortest_path(n,We,graph,start):
    
    adjacent = adjacent_matrix(graph,n)
    
    #Make sure the distance matrix is symmetric
    for i in range(n):
        for j in range(i):
            We[i,j] = We[j,i]
        
    cqm = ConstrainedQuadraticModel()
    
    valid_weight = We*adjacent
    
    y = np.array([[Binary(f"y{i}{j}") if (j!=i) and (j!=start) else 1 for j in range(n)] for i in range(n) if i != start])
    
    path_length = 0
    for i in range(y.shape[0]):
        path_length += 0.5*np.dot(y[i,:],np.dot(valid_weight,y[i,:].transpose()))
    
    cqm.set_objective(path_length)
    
    for i in range(y.shape[0]):
        if i<start:
            end=i
        else:
            end=i+1
        term1 = 0.5*np.dot(y[i,:],np.dot(adjacent,y[i,:].transpose()))
        term2 = quicksum([y[i,k] for k in range(n) if (k!=end) and (k!=start)])
        cqm.add_constraint(term1-term2 == 1)
        
    #end and start has exactly one connection with the chosen nodes
    for i in range(y.shape[0]):
        if i<start:
            end=i
        else:
            end=i+1
        cqm.add_constraint(quicksum([adjacent[start,k]*y[i,k] for k in range(n)]) == 1)
        cqm.add_constraint(quicksum([adjacent[end,k]*y[i,k] for k in range(n)]) == 1)
        
    #each station on the path cannot have more than 2 connections
    for i in range(y.shape[0]):
        if i<start:
            end=i
        else:
            end=i+1
        for j in [k for k in range(n) if (k!=start) and (k!=end)]:
            cqm.add_constraint(quicksum(y[i,j]*adjacent[j,:]*y[i,:]) <= 2)
    
    token = "Insert D Wave API Token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="shortest_path")
    
    data = pd.DataFrame([sampleset.record[i][0] for i in range(len(sampleset.record))],columns=sampleset.variables)
    
    #Take out samples that fulfill the constraints and are connected graph
    feasible_index = np.where(sampleset.record.is_feasible == True)[0]
    optimal_index = np.where(sampleset.record[feasible_index].energy == np.min(sampleset.record[feasible_index].energy))[0]
    optimal_sol = data.iloc[feasible_index[optimal_index][0],:]
    
    
    #Convert the ouput variables into the correct order
    value_dict = dict(optimal_sol) 
    station_list= []
    for i in range(n):
        if i == start:
            station_list.append([i])
        else:
            
            station = [start,i]
            for j in range(n):
                if (j!=i) and (j!=start):
                    if (value_dict[f"y{i}{j}"] == 1):
                        station.append(j)
            station_list.append(station)
    
    path_list=[]
    dist_list=[]
    for i in range(n):
        path = [start]
        unvisit = deepcopy(station_list[i])
        unvisit.remove(start)
        num = len(station_list[i])
    
        for rounds in range(1,num):
            for s in unvisit:
                if adjacent[s,path[-1]] == 1:
                    path.append(s)
                    unvisit.remove(s)
        path_list.append(path)
        
        
        shortest_dist = np.sum([We[path[i],path[i+1]] for i in range(len(path)-1)])
        dist_list.append(shortest_dist)
    
    return [path_list,dist_list]


#Average path length using networkx
def avg_path_length(n,graph_list,W):
    avg_path = []
    #First create a Graph object with networkx
    for graph in graph_list:
        g = nx.Graph()

        for i in range(n):
            g.add_node(i)
        for j in graph:
            g.add_edge(j[0],j[1])

        for i in graph:
            g[i[0]][i[1]]["weight"] = W[i[0]][i[1]]
        
            #calculate the average path length
        avg_path.append(nx.average_shortest_path_length(g,weight="weight"))
        
    return np.array(avg_path)