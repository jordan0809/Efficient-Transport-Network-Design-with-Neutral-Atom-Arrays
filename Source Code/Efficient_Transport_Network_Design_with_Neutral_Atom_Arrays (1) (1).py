#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install pulser


# # Step 0:  Stations Positioning(Maximum Weighted Independent Set)
# 

# In[1]:


import numpy as np
import pulser
from pprint import pprint
from pulser import Pulse, Sequence, Register
import matplotlib.pyplot as plt
from pulser.devices import Chadoq2
from pulser.waveforms import ConstantWaveform, InterpolatedWaveform
from pulser_simulation import QutipEmulator
from pulser.devices import MockDevice
import qutip
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


# In[14]:


#Problem setup of 10 train stations
rows = 2
cols = 5

# Calculate the grid coordinates
x = np.linspace(0, cols - 1, cols)
y = np.linspace(0, rows - 1, rows)
square = np.array([[i, j] for i in x for j in y], dtype=float)

# Adjust the coordinates
square -= np.mean(square, axis=0)
square *= Chadoq2.rydberg_blockade_radius(2.0)-1


qubits = dict(enumerate(square))
reg1 = Register(qubits)


# In[5]:

plt.figure(figsize=(10,6))
reg1.draw(
    blockade_radius=Chadoq2.rydberg_blockade_radius(2.0),
    draw_graph=True,
    draw_half_radius=True,)


# In[15]:


seq = Sequence(reg1, Chadoq2)

seq.declare_channel("local1", "raman_local")
seq.declare_channel("local2", 'rydberg_local')
seq.declare_channel("global", 'rydberg_global')

seq.target(0,"local1")
seq.target(1,"local2")


# In[10]:


N = 10

#define local detuning
W = np.random.randint(1,11,N)
detuning = W

#rescale_detuing (set maximum detuing to be 100 minimum to be 10)
def rescale(w):
    xmin = np.min(w)
    xmax = np.max(w)
    rescaled_detun = [(w[i]-xmin)*(100-10)/(xmax-xmin)+10 for i in range(len(w))]
    return rescaled_detun

rescaled_detuning = rescale(detuning)


# In[11]:


rescaled_detuning


# ## Trotterized Quantum Adiabatic Algorithm (TQAA) implementation for selection of train stations

# In[16]:


#duration of a pulse
duration = 72

#constant rabi pulse for 50ns 2rad/s and 0 detuning
rabi_simple_pulse = Pulse.ConstantPulse(duration, 2, 0, 0)


#a list of constant local detuning pulses
local_simple_pulse = [Pulse.ConstantPulse(duration, 0, rescaled_detuning[i], 0) for i in range(N)]

#Define number of trotterization
iteration = 32

trott = 0
while trott < iteration:
    trott += 1

    #Each trotterization starts with a global rabi drive
    seq.add(rabi_simple_pulse, "global")
    seq.delay(duration,"local1")
    seq.delay(duration,"local2")

    #A series of  local detuing
    for i in range(N//2):
        seq.target(2*i, "local1")
        seq.target(2*i+1, "local2")

        seq.add(local_simple_pulse[2*i], "local1",protocol="no-delay")
        seq.add(local_simple_pulse[2*i+1], "local2",protocol="no-delay")


    if N%2 != 0:
        seq.target(N-1,"local1")
        seq.target(0,"local2")
        seq.add(local_simple_pulse[N-1],"local1",protocol="no-delay")
        seq.add(Pulse.ConstantPulse(duration, 2, 0, 0),"local2")
serialized_sequence = seq.to_abstract_repr()
seq.measure(basis="ground-rydberg")

seq.draw()


# # Step 1:  Partition of Stations to Minimize Overall length

# In[77]:


#Define the QUBO matrix
N = 10
distances = np.random.randint(1, 100, size=(N, N))
np.fill_diagonal(distances, 0)
Wij = distances
alpha = 13

Q = Q = np.zeros((N, N))
for i in range(N):
    for j in range(i+1,N):
        Q[i,j] = Wij[i,j]+alpha
        Q[j,i] = Wij[i,j]+alpha
        
for i in range(N):
    Q[i,i] = alpha*(1-N)-np.sum(Wij[i,:])

Q


# In[96]:


#Now that we have the QUBO instance
bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
costs = []
# this takes exponential time with the dimension of the QUBO
for b in bitstrings:
    z = np.array(list(b), dtype=int)
    cost = z.T @ Q @ z
    costs.append(cost)
zipped = zip(bitstrings, costs)
sort_zipped = sorted(zipped, key=lambda x: x[1])
print(sort_zipped[:10])


# In[97]:


def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = ['1101010100','1101010010' ,'0101110010']  # QUBO solutions
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()


# In[80]:


def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise
    distances are conserved"""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    new_Q = squareform(Chadoq2.interaction_coeff / pdist(new_coords) ** 6)
    return np.linalg.norm(new_Q - Q)


# In[81]:


shape = (len(Q), 2)
costs = []
np.random.seed(0)
x0 = np.random.random(shape).flatten()
res = minimize(
    evaluate_mapping,
    x0,
    args=(Q, shape),
    method="Nelder-Mead",
    tol=1e-6,
    options={"maxiter": 200000, "maxfev": None},
)
coords = np.reshape(res.x, (len(Q), 2))


# In[82]:


qubits = dict(enumerate(coords))
reg = Register(qubits)
plt.figure(figsize=(15,20))
reg.draw(
    blockade_radius=Chadoq2.rydberg_blockade_radius(1.0),
    draw_graph=False,
    draw_half_radius=True,
)


# ## Quantum Adiabatic Algorithm implementation for solving the partition problem

# In[83]:


seq = Sequence(reg, Chadoq2)
seq.declare_channel("global", 'rydberg_global')

N = 10

Omega = 1
delta_0 = -alpha  
delta_f = alpha
T = 8000  # time in ns, we choose a time long enough to ensure the propagation of information in the system

adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),
    0,
)

seq.add(adiabatic_pulse, "global")
seq.draw()


# In[84]:


simul = QutipEmulator.from_sequence(seq)
results = simul.run()
final = results.get_final_state()
count_dict = results.sample_final_state()
plot_distribution(count_dict)


# In[93]:


min_index = np.argmax([item for key,item in count_dict.items()])
[key for key,item in count_dict.items()][min_index]


# In[100]:


x = np.array([1,1,0,1,0,1,0,0,0,0])
H = np.dot(x.transpose(),np.dot(Q,x))
deviation = (H+1840)/1840
print("Qutip Emulator minimum_value:",H)
print("Classical Solver minimum_value:",-1840)
print("deviation:",deviation*100,"%")


# # Step 2: (Classical step) Transfer Frequency Genetic Algorithm

# In[18]:


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


# In[17]:


import imageio as iio
 
# read an image 
img = iio.imread("minimum transfer map.png")
 
plt.figure(figsize=(15,8))
plt.imshow(img)


# In[22]:


import matplotlib.pyplot as plt

# Define optimal lines and iteration
optimal_lines = [[28, 20], [2, 5, 12], [6, 8, 17, 26, 23], [27, 0], [18, 24, 4, 16, 11], [21, 1, 29, 27, 13, 24, 7, 22, 17, 28, 9, 19, 3, 5, 15], [14, 19, 25], [10, 13]]
iteration = 16

# Create a directed graph
G = nx.DiGraph()

# Define colors for each optimal line
line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'gray']

# Add edges based on optimal lines with different colors
for i, line in enumerate(optimal_lines):
    G.add_edges_from(zip(line, line[1:]), color=line_colors[i], linewidth=2, linestyle='dashed')

# Draw the graph
pos = nx.spring_layout(G)
edge_colors = [G[u][v]['color'] for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue', font_size=8, font_color='black', edge_color=edge_colors, width=2, style='dashed')

# Set plot title
plt.title(f'Optimal Lines Visualization (Iteration {iteration})')

# Show the plot
plt.show()


# In[ ]:




