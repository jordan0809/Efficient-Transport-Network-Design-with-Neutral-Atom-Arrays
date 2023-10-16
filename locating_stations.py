# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:35:57 2023

@author: Lai, Chia-Tso
"""

import numpy as np
from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system.samplers import LeapHybridCQMSampler
from dimod.binary import quicksum


#calculate the distance between two nodes on a grid
def calc_dist(x,y,n):  #n is the side length of the square
    return np.sqrt((x%n-y%n)**2+(x//n-y//n)**2)

#Choose location for stations
def choose_location(N,W_loc,D,radius):
    
    cqm = ConstrainedQuadraticModel()
    
    x = np.array([Binary(i) for i in range(N)])
    
    objective = quicksum(-W_loc*x)  #Wanna maximize the chosen weight
    cqm.set_objective(objective)
    
    for i in range(N):
        for j in range(i+1,N):
            if D[i,j] < radius:
                cqm.add_constraint(x[i]*x[j] == 0)
                
    token = "Insert D Wave API token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="choose_location")
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    best_index = np.argmin(feasible_sampleset.record.energy)
    best_sol = feasible_sampleset.record[best_index][0]
    
    chosen_stations = np.where(best_sol == 1)[0]
    
    return chosen_stations