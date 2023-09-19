# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:57:02 2023

@author: Lai, Chia-Tso
"""


import numpy as np
import pandas as pd
from dimod import Binary
import networkx as nx
import random
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler
from dimod.binary import quicksum
from copy import deepcopy
from itertools import groupby


#First divide all the stations into 2 or 2^n groups to reduce problem size
def divide_region(N,We):
    
    cqm = ConstrainedQuadraticModel()
    
    W = np.array([We[j][i] if i>j else We[i][j] for i in range(N) for j in range(N)]).reshape(N,N)
    W[range(N),range(N)] = 0 #diaganol element = 0
    
    x = np.array([Binary(i) for i in range(N)])
    
    objective = 0.5*np.dot(1-x,np.dot(W,(1-x).transpose()))+0.5*np.dot(x,np.dot(W,x.transpose()))
    
    cqm.set_objective(objective)
    
    #cqm.add_constraint(0.5*np.dot(1-x,np.dot(W,(1-x).transpose()))-0.5*np.dot(x,np.dot(W,x.transpose()))>=0)
    
    cqm.add_constraint(quicksum([x[i] for i in range(N)]) == int(N/2))
    
    token = "Insert D Wave Leap API Token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="stations_division")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_sol = feasible_sampleset.record[best_index][0]
    
    group1 = np.where(best_sol == 0)[0]
    group2 = np.where(best_sol == 1)[0]
    
    return [group1,group2]


#make distance matrix from the subgroup nodes
def make_W(W,nodes):
    w = [0 if (i>j) or (i==j) else W[nodes[i]][nodes[j]] for i in range(len(nodes)) for j in range(len(nodes))]
    w = np.array(w).reshape(len(nodes),len(nodes))
    return w



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



#Make the adjacent matrix of a graph
def adjacent_matrix(graph,n):

    adjacent = np.zeros((n,n))
    for edge in graph:
        adjacent[edge[0],edge[1]] = 1
        adjacent[edge[1],edge[0]] = 1
    
    return adjacent



#Shortest path algorithm with quantum annealing
def shortest_path(n,We,graph,adjacent,start,end):
    
    if adjacent[start,end] == 1:
        station = [start,end]
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
        
    
    return [path,shortest_dist]



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


#make a nx.Graph object from a list of edges(graph)
def make_nx_graph(n,graph,W,nodes):  
    
    g = nx.Graph()
    for i in nodes:
        g.add_node(i)
    for j in graph:
        g.add_edge(j[0],j[1])
    
    for i in graph:
        g[i[0]][i[1]]["weight"] = W[i[0]][i[1]]
    return g 


#pick out the best network from step2 and turn it into a nx.Graph object
def best_network(n,graph_list,W):
    index = np.argsort(overall_length(graph_list,W)*avg_path_length(n,graph_list,W))[0]
    
    g = nx.Graph()
    best = graph_list[index]

    for i in range(n):
        g.add_node(i)
    for j in best:
        g.add_edge(j[0],j[1])

    for i in best:
        g[i[0]][i[1]]["weight"] = W[i[0]][i[1]]
        
    return g



#Now connect the subnetworks divided from step 1 back together
def connection(n_sub,nx_graph_list,size_sub,W):  #n_sub: number of subnetworks, size_sub: # of nodes in each subnetwork
    new_connection = []
    for k in range(n_sub):  
        for s in range(k+1,n_sub):
            avg =[]
            for i in list(nx_graph_list[k].nodes):
                for j in list(nx_graph_list[s].nodes):
                    F12 = nx.union(nx_graph_list[k],nx_graph_list[s])
                    F12.add_edge(i,j,weight=W[i,j])
                    avg.append(nx.average_shortest_path_length(F12,weight="weight"))
            mini = np.argmin(avg)
            con1 = list(nx_graph_list[k].nodes)[mini//size_sub]
            con2 = list(nx_graph_list[s].nodes)[mini% size_sub]
            new_connection.append((con1,con2))
    
    g_final = nx.union_all(nx_graph_list)
    weights = [W[i,j] for i,j in new_connection]
    for index,edge in enumerate(new_connection):
        g_final.add_edge(edge[0],edge[1],weight=weights[index])
        
    return g_final




#Assignment of lines to minimize transfer time and overhead in overall length of connections
#Using genetic algorithm for optimization
def generate_population(n,nx_graph,popu_size):
    #create a list of odd connection nodes
    terminal = [i for i in range(n) if np.sum(adjacent_matrix(list(nx_graph.edges),n)[i,:])%2 ==1]
    # shuffle the nodes randomly to form different lines
    population = [random.sample(terminal,len(terminal)) for i in range(popu_size)]
    return population


#Transfer times from a starting station to an ending station
def transfer_freq(nx_graph,chromo,start,end):
    
    lines = [set(nx.shortest_path(nx_graph,chromo[2*i],chromo[2*i+1])) for i in range(int(len(chromo)/2))]
    nl = len(lines)

    short = set(nx.shortest_path(nx_graph,start,end))

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
    end_line = set([i for i in range(nl) if end in lines[i]]) #lines that go through end station

    if len(start_line.intersection(end_line)) !=0: #if any line goes through both start and end then no need to transfer
        transfer  = 0
    else:
        possible =[] #possible transfer combination for each starting line i
        for i in list(start_line):
            route=[i]  #history of line itenary
            current=i  #current line one is at
            uni = set().union(*[lines[k] for k in route])  #the union of all the valid lines explored so far
            while short.issubset(uni) == False:   #when the union has not convered the shortest path yet, continue
                neighbors = np.where(ad[current,:] == 1)[0]  #neighboring lines of current line
                crossing = [uni.union(lines[j]) for j in neighbors]   #try out the union with neighboring lines
                num_intersect = [len(s.intersection(short)) for s in crossing] #number of intersecting elements with short
                best = neighbors[np.argmax(num_intersect)]  #take out the line with the most intersecting elements
                route.append(best)
                current = best
                uni = set().union(*[lines[k] for k in route])
                if np.max(num_intersect) == len(short):   #if every station on the shortest path is coverd, stop
                    break
            possible.append(route)
        transfer = np.min([len(r)-1 for r in possible]) #shortest itenary -1 is the trnasfer time
    
    return transfer


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
        fitness = np.mean([transfer_freq(nx_graph,chromo,i,j) for i in range(n) for j in range(i+1,n)])
    
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


#Assign lines by equalizing the passenger load of each line while forbidding adding extra connections 
def train_line_equal_load(n,Wn,nx_graph):  #Wn here is the weight of the station
    
    cqm = ConstrainedQuadraticModel()
    
    A = adjacent_matrix(list(nx_graph.edges),n)
    connections = np.array([np.sum(A[i,:]) for i in range(n)]) #number of connections on each station
    wn = Wn/connections # the reduced load of each station
    
    trans_points = [i for i in range(n) if np.sum(A[i,:]) >= 3]
    non_trans = [i for i in range(n) if np.sum(A[i,:]) <= 2]
    nl = len(trans_points)+1 #number of lines
    
    terminal = [i for i in range(n) if np.sum(A[i,:]) in [1,3]] #terminal stations
    
    x = np.array([Binary(f"x{i}{j}") for i in range(nl) for j in range(n)]).reshape(nl,n)
    
    #minimize the difference between sums of any two lines
    objective = 0
    for k in range(nl):
        for s in range(k+1,nl):
            objective += (quicksum([x[k][j]*wn[j] for j in range(n)])-quicksum([x[s][j]*wn[j] for j in range(n)]))**2
    cqm.set_objective(objective)
    
    #need to form a path (branch roads and crossroads are also included here)
    for i in range(nl):
        term1 = 0.5*np.dot(x[i,:],np.dot(A,x[i,:].transpose()))
        term2 = quicksum([x[i,k] for k in range(n)])
        cqm.add_constraint(term2-term1 == 1)
    
    #each line cannot have any station with more than 2 connections
    for i in range(nl):
        for j in range(n):
            cqm.add_constraint(quicksum(x[i,j]*A[j,:]*x[i,:]) <= 2) #xij*Aj is the adjacent of "chosen" node
    
    
    #Each line needs to contain at least 2 terminal stations (3-connection stations are both terminal and intermediate)
    for i in range(nl):
        cqm.add_constraint(quicksum([x[i,k] for k in terminal]) >= 2)
    
    #each connected pair of stations need to be on at least one same line
    for j in range(n):
        for k in range(j+1,n):
            if A[j,k] == 1:
                cqm.add_constraint(quicksum([x[i,j]*x[i,k] for i in range(nl)]) >= 1)
        
    #every station needs to be covered
    for j in range(n):
        cqm.add_constraint(quicksum([x[k,j] for k in range(nl)]) >= 1)
    #transfer points need to have at exactly 2 lines going through
    for j in trans_points:
        cqm.add_constraint(quicksum([x[k,j] for k in range(nl)]) ==2)
    #non_transfer points need to have exactly 1 line going through
    for j in non_trans:
        cqm.add_constraint(quicksum([x[k,j] for k in range(nl)]) ==1)
    
    token = "Insert D Wave API token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="stations_division")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    data = pd.DataFrame(feasible_sampleset)
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_dict = dict(data.iloc[best_index,:])
    
    line_list=[]
    for i in range(nl):
        line = []
        for j in range(n):
            if best_dict[f"x{i}{j}"] == 1:
                line.append(j)
        line_list.append(line)
    
    return [line_list,feasible_sampleset]



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
    
    return [lines,feasible_sampleset]

    