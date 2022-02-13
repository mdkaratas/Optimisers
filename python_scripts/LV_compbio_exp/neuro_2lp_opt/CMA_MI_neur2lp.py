#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:49:08 2022

@author: melikedila
"""

import matlab.engine
import numpy as np
import sys
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
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro2lp_costfcn"
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

def neuro2lp_gates(inputparams):
    init = inputparams
    inputparams = inputparams[0:8]
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = init[8:14]
    gates = list(gates)
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

def neuro2lp(inputparams):
    for i in inputparams:
        if (inputparams[0] + inputparams[2] < 24) :
            if (inputparams[1] + inputparams[3] < 24):
                inputparams[8] = round(inputparams[8])
                inputparams[9] = round(inputparams[9])
                inputparams[10] = round(inputparams[10])
                inputparams[11] = round(inputparams[11])
                inputparams[12] = round(inputparams[12])
                cost=neuro2lp_gates(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24
                cost = dist + neuro2lp_gates(inputparams)
        else:
            if (inputparams[1] + inputparams[3] < 24):
                dist = (inputparams[0] + inputparams[2] - 24)
                cost = dist + neuro2lp_gates(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                cost = dist + neuro2lp_gates(inputparams)
    return cost     #inputparams = [12.0,6.1,3.5,0.4,0.6,0.04,0.7]


cma_lb = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,24,24,12,1,1,1,1,1,1,1,1] ,dtype='float')

max_fevals = 5*10**3
d = 5  # bits in logic gate
n_pop = 4+(3* (np.log(d))) #6.079
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}


X = []  # list of solutions 
for i in range(1):
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    z = random.uniform(0,24)
    while x+z > 24 :
        x = random.uniform(0,24)
        z = random.uniform(0,24)  
    t = random.uniform(0,24)
    while y+t > 24 :
        y = random.uniform(0,24)
        t = random.uniform(0,24)    
    u = np.random.uniform(0,12) 
    v = np.random.uniform(0,1) 
    p = np.random.uniform(0,1)
    w = np.random.uniform(0,1)
    v1 = np.random.uniform(0,1)
    v2 = np.random.uniform(0,1)
    v3 = np.random.uniform(0,1)
    v4 = np.random.uniform(0,1)
    v5 = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,v,p,w,v1,v2,v3,v4,v5] 
    init_sigma = 0.5
    es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
    sol = es.optimize(neuro2lp).result 
    X.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store I


with open(f"Desktop/x_CMAES_MI.txt", "wb") as fp:   #Pickling
 pickle.dump(X, fp)   
with open(f"Desktop/x_CMAES_MI.txt", "rb") as fp:   # Unpickling
 X = pickle.load(fp)

F = [] 
for i in X:
    F.append(neuro2lp(i))   
# store init F    
with open(f"Desktop/f_CMAES_MI.txt", "wb") as fp:   #Pickling
 pickle.dump(F, fp)   
with open(f"Desktop/f_CMAES_MI.txt", "rb") as fp:   # Unpickling
 f_CMA_1 = pickle.load(fp)





   
















