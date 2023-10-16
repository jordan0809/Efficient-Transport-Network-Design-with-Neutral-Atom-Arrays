# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:46:20 2023

@author: Lai, Chia-Tso
"""

import numpy as np
from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler
from dimod.binary import quicksum


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