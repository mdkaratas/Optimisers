#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:11:10 2022

@author: melikedila
"""

import matlab.engine
import numpy as np
import cma
import pickle
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem
from pynmmso.listeners import TraceListener
from pyDOE import *
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import itertools


eng = matlab.engine.start_matlab()


#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataDD.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingDD.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataDD']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingDD']

#%%

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi
gate = gatesm[1]




# Cost func. - node update

num_T = 2   # number of thresholds
samples = 10
Threshs = lhs(num_T, samples)
Threshs = list(Threshs)
#with open("Desktop/Threshs_av_thresh_0.5.txt", "wb") as fp:   
# pickle.dump(Threshs, fp)  
def neuro1lp_costf(inputparams):
    #print(inputparams[3])
    agg_cost =[]
    agg_x =[]
    init = inputparams
    gates = gate
    gates = list(gates)
    gates = matlab.double([gates])
    for id in range(len(Threshs)):
        inputparams = init
        inputparams[3] = float(Threshs[id][0] ) # change thresholds here 
        inputparams[4] = float(Threshs[id][1] )
        agg_x.append(inputparams)
        inputparams = inputparams[0:5]
        inputparams = list(inputparams)    
        inputparams = matlab.double([inputparams])       
        agg_cost.append(eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1))
    min_val = min(agg_cost)
    min_index = agg_cost.index(min_val)   
    avr_cost = np.mean(agg_cost) 
    return avr_cost

cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1] ,dtype='float')
max_fevals = 5*10**3
d = 2   # bits in logic gate
n_pop = 4+(3* (np.log(d))) #6.079
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}
def neuro1lp(inputparams):
    for i in inputparams:
        if inputparams[0] + inputparams[1] < 24:
           cost=neuro1lp_costf(inputparams)
        else:
            dist = inputparams[0] + inputparams[1] - 24
            cost = dist + neuro1lp_costf(inputparams)
    return cost

###########################################################################################
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



#%%

# NM -- Find LO through NM runs

nonzdelt = 0.05
zdelt = 0.00025/0.05
ndim = 5
st = {}
threshold = 10e-3
fvals = []
xvals = []
permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 10
for j in range(rep):
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    #k = np.random.uniform(0,1)
    #l = np.random.uniform(0,1)
    #temp_xvals= []
    #temp_fvals = []
    I = {}
    t = random.uniform(0,1)
    u = random.uniform(0,1)
    #init_sol = [x,y,z,t,u,k,l] 
    init_sol = [x,y,z,t,u] 
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = np.array(init_sol)
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[0].reshape(-1,1)* step
    #I[id] = init_simplex[0]
    OptimizeResult = minimize(neuro1lp, init_simplex[0], method='nelder-mead', 
                          options={'initial_simplex': init_simplex, 'xatol': 1e-8, 'disp': False})
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    #min_val = min(temp_fvals)
    #min_index = temp_fvals.index(min_val)     
    #xvals.append(temp_xvals[min_index]) 
    #fvals.append(np.mean(temp_fvals))   
    st[j] = init_simplex
    #st[j] = init_simplex
    print("--- %s seconds ---" % (time.time() - start_time))  
st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list]



with open("Desktop/nodes_list.txt", "wb") as fp:   
 pickle.dump(nodes_list, fp)  
with open(f"Desktop/nodes_list.txt", "rb") as fp:   
 nodes_list = pickle.load(fp)   
    
with open("Desktop/fvals_list.txt", "wb") as fp:   
 pickle.dump(fvals_list, fp)   
with open("Desktop/fvals_list.txt", "rb") as fp:   
 fvals_list = pickle.load(fp)     
 
 
 
dist_nodes = [] 
for i in range(len(nodes_list)):
    for j in range(i+1,len(nodes_list)):
        print(nodes_list[i][0:3],nodes_list[j][0:3])
        dist_nodes.append(euclidean_distances([nodes_list[i][0:3]],[nodes_list[j][0:3]]))
       
tot_dist = np.mean(dist_nodes)    
A = (2* tot_dist )  
B = len(nodes_list) * (len(nodes_list) - 1)
avr_dist = A / B

with open("Desktop/avr_dist.txt", "wb") as fp:   
 pickle.dump(avr_dist, fp)   
with open("Desktop/avr_dist.txt", "rb") as fp:   
 avr_dist = pickle.load(fp)    


###############################

global_edges = [] 
nonzdelt = 0.65
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)): 
    start_time = time.time()
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(neuro1lp, nodes_list[p], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': True})
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)

    edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique
    local_edges = [(p, idx, count[j]* 1/(2**ndim)) for j,idx in enumerate(uniqe_edge_indices)] #* 1/((2**ndim)*len(nodes_list))
    nm= np.array(local_edges)
    nm = nm[:,2]
    norm = [float(i)/sum(nm) for i in nm]
    for e,w in enumerate(local_edges):
        w = list(w)
        w[2]= norm[e]
        w= tuple(w)
        local_edges[e]=w
        print(local_edges)    
    global_edges = global_edges + local_edges
    print("--- %s seconds ---" % (time.time() - start_time))      









