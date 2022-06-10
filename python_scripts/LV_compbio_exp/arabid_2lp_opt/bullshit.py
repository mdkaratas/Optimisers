#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:59:32 2022

@author: mkaratas
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
from joblib import Parallel, delayed
import multiprocessing


eng = matlab.engine.start_matlab()



# Set required paths
root = '/Users/mkaratas/Desktop/GitHub/'

path= root + r"/BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= path= root + r"/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')['dataLD']
dataDD = eng.load('dataLL.mat')['dataLL']
lightForcingLD = eng.load('lightForcingLD.mat')['lightForcingLD']
lightForcingDD = eng.load('lightForcingLL.mat')['lightForcingLL']




#%%
n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))



Threshs = [0.202582,0.358257,0.512238,0.200218]
#agg_x =[] 
#%%

def arabid2lp_costf(inputparams):
    gates = gate
    gates = matlab.double([list(gates)])
    #agg_x.append(inputparams)
    l = inputparams[9:11]
    inputparams_ = inputparams[0:9]
    inputparams_ = list(inputparams_)
    inputparams_.append(float(Threshs[0])) # add thresholds here 
    inputparams_.append(float(Threshs[1]))
    inputparams_.append(float(Threshs[2]))
    inputparams_.append(float(Threshs[3]))
    inputparams_.append(l[0])
    inputparams_.append(l[1])
    inputparams_ = matlab.double([inputparams_])
    avr_cost = eng.getBoolCost_cts_arabid2lp(inputparams_,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
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

#%%
# NM -- Find LO through NM runs

rep = 100
ndim =11
init_sol = np.empty([rep,ndim])

gate = gatesm[137]
savename = f"{gate}"
nonzdelt = 0.5
zdelt = 0.00025/0.5
st = {}
threshold = 10e-1
permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1
permut = []



with open("/Users/mkaratas/Desktop/actual_LON/init_nodes_list_arab2lp_%s_100s_n_ent.txt"%gate, "rb") as fp:
 nodes_list = pickle.load(fp)

with open("/Users/mkaratas/Desktop/actual_LON/init_fvals_list_arab2lp_%s_100s_n_ent.txt"%gate, "rb") as fp:
 fvals_list = pickle.load(fp)

with open("/Users/mkaratas/Desktop/actual_LON/st_list_arab2lp_%s_100s_n_ent.txt"%gate, "rb") as fp:
 st_list = pickle.load(fp)


#%%



global_edges = []
nonzdelt = 0.5
step = np.eye(ndim) * nonzdelt
start_time = time.time()
for p in range(len(nodes_list)):
    start_time = time.time()
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    par_start_time = time.time()
    simplex_subs = init_simplex[0] + (init_simplex[0] * np.expand_dims(permut, axis=-1)) * step
    simplex0 = np.repeat(np.expand_dims(np.expand_dims(init_simplex[0], axis=0), axis=0), len(permut), axis=0)
    init_simplices = np.concatenate((simplex0, simplex_subs), axis=1)


    all_args = [(arabid2lp, init_simplex[0], {'method':'nelder-mead', 'options':{'initial_simplex': simplex, 'xatol': 1e-4,'fatol': 1e-3}}) for simplex in init_simplices]
    num_cores = min(multiprocessing.cpu_count(), 32)
    OptimizeResults = Parallel(n_jobs=num_cores, backend="threading")(delayed(minimize)(*args, **kwargs) for *args, kwargs in all_args)
    xvals_post = [result.x for result in OptimizeResults]
    fvals_post = [result.fun for result in OptimizeResults]
    end_time = time.time()
    print('edge for %s th lo: '%p, time.time() - end_time)

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




with open("actual_LONs/nodes_list_ara2lp_{savename}_100s%s.txt"%gate, "wb") as fp:
 pickle.dump(nodes_list, fp)

with open("actual_LONs/global_edges_ara2lp_{savename}_100s%s.txt"%gate, "wb") as fp:
 pickle.dump(global_edges, fp)

with open("actual_LONs/fvals_list_ara2lp_{savename}_100s%s.txt"%gate, "wb") as fp:
 pickle.dump(fvals_list, fp)















