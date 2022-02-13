#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:30:53 2022

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

#####################################################      CMA-ES: G x G  01

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi
gate = gatesm[1]

def neuro1lp_costf(inputparams):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = gate
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

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


 # I initial pts used
x_CMA_1 = [] # list of solutions 
for i in range(3):
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    u = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u] 
    init_sigma = 0.5
    es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
    sol = es.optimize(neuro1lp).result 
    x_CMA_1.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store X
with open("Desktop/x_CMA_1.txt", "wb") as fp:   #Pickling
    pickle.dump(x_CMA_1, fp)   
with open("Desktop/x_CMA_1.txt", "rb") as fp:   # Unpickling
    x_CMA_1 = pickle.load(fp)

f_CMA_1 = [] 
for i in x_CMA_1:
    f_CMA_1.append(neuro1lp_gates(i,[0,1]))   
                   
# store init X
with open("Desktop/f_CMA_1.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_1 , fp)   
with open("Desktop/f_CMA_1.txt", "rb") as fp:   # Unpickling
    f_CMA_1 = pickle.load(fp)