#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 20:51:22 2022

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
root = '/Users/melikedila/Documents/GitHub/'
path= root + r"/BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataLL.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingLL.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataLL']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingLL']

#%%
n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi


# Cost for average across thresholds

num_T = 4   # number of thresholds
samples_T = 10
Threshs = lhs(num_T, samples=samples_T)
Threshs = list(Threshs)
#agg_x =[] 
#%%

def arabid2lp_costf(inputparams):
    #print(inputparams[3])
    agg_cost =[]
    gates = gate
    gates = list(gates)
    gates = matlab.double([gates])
    #agg_x.append(inputparams)
    l = inputparams[9:11]
    inputparams_ = inputparams[0:9]
    inputparams_ = list(inputparams_)
    for id in range(len(Threshs)):
        inputparams_.append(float(Threshs[id][0])) # add thresholds here 
        inputparams_.append(float(Threshs[id][1]))
        inputparams_.append(float(Threshs[id][2]))
        inputparams_.append(float(Threshs[id][3]))
        inputparams_.append(l[0])
        inputparams_.append(l[1])
        inp = np.array(inputparams_)
        inputparams_ = matlab.double([inputparams_])
        agg_cost.append(eng.getBoolCost_cts_arabid2lp(inputparams_,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1))
        inputparams_ = list(inp)
    #min_val = min(agg_cost)
    #min_index = agg_cost.index(min_val)   
    avr_cost = np.mean(agg_cost)
    return avr_cost

def arabid2lp(inputparams):
    for i in inputparams:
        if (inputparams[0] + inputparams[2] < 24) :
            if (inputparams[1] + inputparams[3] < 24):
                cost=arabid2lp_costf(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24
                cost = dist + arabid2lp_costf(inputparams)
        else:
            if (inputparams[1] + inputparams[3] < 24):
                dist = (inputparams[0] + inputparams[2] - 24)
                cost = dist + arabid2lp_costf(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                cost = dist + arabid2lp_costf(inputparams)
    return cost



#%%

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


#euclidean_distances((nodes_list[2], nodes_list[3]))


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

#%%
# NM -- Find LO through NM runs
for i in range(len(152,156):
    for count in range(30):           
               
        gate = gatesm[i]
        savename = f"{gate}"
        nonzdelt = 0.5
        zdelt = 0.00025/0.5
        ndim =11
        st = {}
        threshold = 10e-1
        fvals = []
        xvals = []
        permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
        permutations[permutations==0] = -1
        step = np.eye(ndim) * nonzdelt
        #p ={}
        
        rep = 100
        #start = {}
        for j in range(rep):
            start_time = time.time()
            for i in range(1,7):
                globals()['x_%s' % i]  = random.uniform(0,24)
            for i in range(7,10):
                globals()['x_%s' % i]  = random.uniform(0,12)      
            for i in range(10,12):
                globals()['x_%s' % i]  = random.uniform(0,4)  
            while any([ x_1+ x_2 +x_3 >= 24, x_5 + x_6 >= 24, x_2 + x_3 + x_4 + x_6 >= 24 ]):
                for i in range(1,7):
                    globals()['x_%s' % i]  = random.uniform(0,24)
                for i in range(7,10):
                    globals()['x_%s' % i]  = random.uniform(0,12)    
                for i in range(10,12):
                    globals()['x_%s' % i]  = random.uniform(0,4)  
        
            init_sol = [globals()['x_%s' % i] for i in range(1,12) ] 
            #start[j]= init_sol
            init_simplex = np.zeros((ndim+1, ndim))
            init_simplex[0] = np.array(init_sol)
            init_simplex[init_simplex==0] = zdelt
            init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[0].reshape(-1,1)* step
            #I[id] = init_simplex[0]
            OptimizeResult = minimize(arabid2lp, init_simplex[0], method='nelder-mead',
                                  options={'initial_simplex': init_simplex, 'xatol': 1e-8, 'disp': False})
            xvals.append(OptimizeResult.x)
            fvals.append(OptimizeResult.fun)
            st[j] = init_simplex
            #st[j] = init_simplex
            print("--- %s seconds ---" % (time.time() - start_time))
        st = optima_merge(xvals, st, threshold)
        #print('st: ', st)
        
        # create the initial list of nodes
        nodes_list = [xvals[k] for k,v in st.items()]
        fvals_list = [fvals[k] for k,v in st.items()]
        fvals_list = [round(num, 12) for num in fvals_list]
        st_list= []
        for i,k in st.items():
            st_list.append(len(k))
        opt = np.column_stack((fvals_list, nodes_list,st_list))
        opt = opt[np.argsort(opt[:, 0])]
        fvals_list = opt[:,0]
        nodes_list = opt[:,1:ndim+1]
        st_list = opt[:,ndim+1]
        fvals_list = list(fvals_list)
        nodes_list = list(nodes_list)
        st_list = list(st_list)
        
        
        with open("actual_LONs/prev_nodes_list_ara2lp_{savename}_100s%s.txt"%count, "wb") as fp:
         pickle.dump(nodes_list, fp)
        
        with open("actual_LONs/st_list_ara2lp_{savename}_100s%s.txt"%count, "wb") as fp:
         pickle.dump(global_edges, fp)
        
        #%%
        
        
        
        global_edges = []
        nonzdelt = 0.5
        step = np.eye(ndim) * nonzdelt
        for p in range(len(nodes_list)):
            start_time = time.time()
            xvals_post = []
            fvals_post = []
            init_simplex = np.zeros((ndim+1, ndim))
            init_simplex[0] = nodes_list[p]
            for i in range(2**ndim):
                init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
                OptimizeResult = minimize(arabid2lp, nodes_list[p], method='nelder-mead',
                                          options={'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13})
                xvals_post.append(OptimizeResult.x)
                fvals_post.append(OptimizeResult.fun)
        
            edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
            uniqe_edge_indices = list(set(edge_indices))
            count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique
            local_edges = [(p, idx, count[j]* 1/(2**ndim)) for j,idx in enumerate(uniqe_edge_indices)] #* 1/((2**ndim)*len(nodes_list))
            nm= np.array(local_edges)
            weight = nm[:,2]
            norm_weight = [float(l)/sum(weight) for l in weight]
            for e,w in enumerate(local_edges):
                w = list(w)
                w[2]= norm_weight[e]
                w= tuple(w)
                local_edges[e]=w
                print(local_edges)
            global_edges = global_edges + local_edges
            print("--- %s seconds ---" % (time.time() - start_time))
        
        
        print('nodes_list length is: ', len(nodes_list))
        print('created edges: ', global_edges)
        print('fvals: ', fvals_list)
        
        
        
        
        with open("actual_LONs/nodes_list_ara2lp_{savename}_100s%s.txt"%count, "wb") as fp:
         pickle.dump(nodes_list, fp)
        
        with open("actual_LONs/global_edges_ara2lp_{savename}_100s%s.txt"%count, "wb") as fp:
         pickle.dump(global_edges, fp)
        
        with open("actual_LONs/fvals_list_ara2lp_{savename}_100s%s.txt"%count, "wb") as fp:
         pickle.dump(fvals_list, fp)
        