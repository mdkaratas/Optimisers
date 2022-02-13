#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:38:17 2022

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
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
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

def arabid2lp_gates(inputparams):
    init = inputparams
    inputparams = inputparams[0:15]
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = init[15:23]
    gates = list(gates)
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_arabid2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

def arabid2lp(inputparams):
    dist = []
    if (inputparams[0] + inputparams[1] + inputparams[2] >= 24) :
        dist.append(inputparams[0] + inputparams[1] + inputparams[2] - 23.5)
        
    if inputparams[4] + inputparams[5] >= 24:
        dist.append(inputparams[4] + inputparams[5] - 23.5)
        
    if inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] >= 24:
        dist.append(inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] - 23.5)  
    
    distance = sum(dist)
    for i in range(15,23):
        inputparams[i] = round(inputparams[i])
    #print(inputparams)    
    #cost=arabid2lp_gates(inputparams)
    cost = arabid2lp_gates(inputparams) + distance
        
    return cost     #
#inputparams = [12.0,6.1,3.5,0.4,0.6,0.04,0.7,6.1,3.5,0.4,0.6,0.04,0.7,0.4,0.6,0.04,0.7,6.1,3.5,0.4,0.6,0.3,0.1]



cma_lb = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,24,24,24,24,12,12,12,1,1,1,1,4,4,1,1,1,1,1,1,1,1] ,dtype='float')

max_fevals = 5*10**3
d = 8  # bits in logic gate
n_pop = 4+(3* (np.log(d))) #6.079
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}

c = 0
X = []  # list of solutions 
while c <=30:
#for i in range(30):    
    start_time = time.time()
    for i in range(1,7):
        globals()['x_%s' % i]  = random.uniform(0,24)
    for i in range(7,10):
        globals()['x_%s' % i]  = random.uniform(0,12)    
    for i in range(10,14):
        globals()['x_%s' % i]  = random.uniform(0,1)    
    for i in range(14,16):
        globals()['x_%s' % i]  = random.uniform(0,4) 
    for i in range(16,24):
        globals()['x_%s' % i]  = random.uniform(0,1)      
    while any([ x_1+ x_2 +x_3 >= 24, x_5 + x_6 >= 24, x_2 + x_3 + x_4 + x_6 >= 24 ]):
        for i in range(1,7):
            globals()['x_%s' % i]  = random.uniform(0,24)
        for i in range(7,10):
            globals()['x_%s' % i]  = random.uniform(0,12)    
        for i in range(10,14):
            globals()['x_%s' % i]  = random.uniform(0,1)    
        for i in range(14,16):
            globals()['x_%s' % i]  = random.uniform(0,4)  

    init_sol = [globals()['x_%s' % i] for i in range(1,24) ] 
    init_sigma = 0.5
    es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
    sol = es.optimize(arabid2lp).result 
    X.append(sol.xbest.tolist())
    c = c + 1
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store I


with open(f"Desktop/x_CMAES_MI_cts_arabid2lp.txt", "wb") as fp:   #Pickling
 pickle.dump(X, fp)   
with open(f"Desktop/x_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   # Unpickling
 X = pickle.load(fp)

F = [] 
for i in X:
    F.append(arabid2lp(i))   
# store init F    
with open(f"Desktop/f_CMAES_MI_cts_arabid2lp.txt", "wb") as fp:   #Pickling
 pickle.dump(F, fp)   
with open(f"Desktop/f_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   # Unpickling
 f_CMA_1 = pickle.load(fp)