# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:59:35 2020

@author: mk633
"""

import numpy as np
import time
import math
import sys
from scipy.optimize import minimize
from pyDOE import *
from scipy.stats.distributions import norm
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph

##############################################################################
# bemchmark functions

def ackley(x):
    return -20* np.exp(-0.2 * np.sqrt( (1/x.shape[0]) * np.sum(x**2) ) ) - \
            np.exp( (1/x.shape[0]) * np.sum( np.cos(2*np.pi*x)) ) + 20 + np.e
    
def rastrigin(x):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * np.cos( 2 * math.pi * x ))

def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)

#def camel_back(x):
#    return 

def optima_merge(keys, basin, threshold):
    merged_basin = {}
    merged_bool = [False]* len(basin.keys())
    distances = euclidean_distances(keys, keys)
    #condition1 =  distances!=0 
    #condition2 = distances<=threshold
    #condition = condition1 & condition2    #### booleans
    condition = distances<=threshold
    for i, row in enumerate(condition):                                # ith row is already checked so continue jth onwards
        #i is the index of each row(whole matrix), j is column of boolean , elem is elem itself in that row (whole matrix)
        # optimally, we only have to check half of the condition matrix (upper or lower parts around the diagnoal part)
        indices = [j for j, elem in enumerate(row) if (elem and j>i)]    
        if(len(indices)!=0):
            merged_bool[i] = True
            merge = False
            for idx in indices:
                #basin[i] = np.vstack((basin[i], basin[idx])) 
                if(not merged_bool[idx]):
                    merge = True
                    basin[i] = np.vstack((basin[i], basin[idx]))
                    merged_bool[idx] = True
              
            if(merge):
                merged_basin[i] = basin[i]
        else:
            # check whether the key at 'i' was already merged above in comparisons with other keys
            if(not merged_bool[i]):  
                merged_basin[i] = basin[i]
            
    return merged_basin
        
def update_nodes(xvals_list, fvals_post , fvals_list, nodes_list, threshold):
    edge_indices = []
    xvals = np.array(xvals_list)
    nodes = np.array(nodes_list)
    dist = euclidean_distances(xvals, nodes)
    dist_bool = dist<=threshold
    for i, row in enumerate(dist_bool):
        idx = np.where(row)[0]
        # if this LO is not in the current nodes list
        if(idx.size==0):
           nodes_list.append(xvals[i])
           fvals_list.append(fvals_post[i])
           edge_indices.append(len(nodes_list)-1)
        # if this LO was already found and strored inside nodes list
        else:
           edge_indices.append(idx[0])
           
    return edge_indices
    


############################################################################

nonzdelt = 0.05
zdelt = 0.00025/0.05
ndim = 2
n = 1
k = 1
st = {}
threshold = 10e-3
fvals = []
xvals = []
permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 100
for j in range(rep):
    x0_cam_0 = np.random.uniform(low=-1.9, high=1.9, size=(k, n))
    x0_cam_1 = np.random.uniform(low=-1.1, high=1.1, size=(k, n))
    init_simplex = np.zeros((ndim+1, ndim))
    random_pick = np.arange(2**ndim)
    np.random.shuffle(random_pick)
    i = random_pick[0]
    init_simplex[0] = np.array([x0_cam_0, x0_cam_1]).ravel()
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[i].reshape(-1,1)* step
    OptimizeResult = minimize(camel_back, init_simplex[0], method='nelder-mead', 
                              options={'initial_simplex': init_simplex, 'xatol': 1e-8, 'disp': False})
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    st[j] = init_simplex
    
st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
#print(nodes_list)

###########################################################################
# scan all initial points selected randomly above and find LOs from each

fvals_post = []
global_edges = []
init_simplex = np.zeros((ndim+1, ndim))
c = 0
for k,v in st.items():
    xvals_post = []
    # pick the same initial point we previously started from for this key
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = v[0]
    init_simplex[init_simplex==0] = zdelt
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(camel_back, v[0], method='nelder-mead', 
                                  options={'initial_simplex': init_simplex, 'xatol': 1e-8, 'disp': False})
        
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)

    # update the nodes list and return existing edge indices
    edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] 

    local_edges = [(c, idx, count[j] * 1/2**ndim) for j, idx in enumerate(uniqe_edge_indices)]
    global_edges = global_edges + local_edges
    c+=1


print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)

##########################################################################
# draw the network

plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')

G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
node_sz = [v.shape[0]*100 for k,v in st.items()]
print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])


# use fitness values as node's names
#fvals_round = np.round(fvals_list, 3)
#G.add_nodes_from(fvals_round)
#G.add_edges_from([[fvals_round[tup[0]], fvals_round[tup[1]]] for tup in global_edges])


# rename nodes
fvals_round = list(np.round(fvals_list, 3))
print(fvals_round)
fvals_names = []
charac = 97
for v in fvals_round:
    fvals_names.append(str(v)+'_%s'%(chr(charac)))
    charac+=1
    
mapping = {k:fvals_names[k] for k in range(len(nodes_list)) }
print('mapping', mapping)
nx.relabel_nodes(G, mapping, copy=False)

nx.draw_networkx(G, with_label = True, node_size=node_sz, width=[tup[2] for tup in global_edges], \
                 node_color =node_color, connectionstyle='arc3, rad = 0.3')
    
A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('cam_benchmark.png')


#########################################################################