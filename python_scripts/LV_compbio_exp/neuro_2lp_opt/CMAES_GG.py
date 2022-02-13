#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:21:25 2021

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

#gate = sys.argv[1]

##  all gate combinations of length of n
n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi
#  lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
#gate = gates[gate-1]

#gate1 = sys.argv[1]
#gate2 = sys.argv[2]
#gate3 = sys.argv[3]
#gate4 = sys.argv[4]
#gate5 = sys.argv[5]

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
for k in range(13,14):
    gate = gatesm[k]
    c = 0
    def neuro2lp_gates(inputparams):
        inputparams = list(inputparams)
        inputparams = matlab.double([inputparams])
        gates = gate
        gates= matlab.double([gates])
        cost=eng.getBoolCost_neuro2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        return cost
    
    def neuro2lp(inputparams):
        for i in inputparams:
            if (inputparams[0] + inputparams[2] < 24) :
                if (inputparams[1] + inputparams[3] < 24):
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
   
    cma_lb = np.array([0,0,0,0,0,0,0,0], dtype='float')
    cma_ub = np.array([24,24,24,24,12,1,1,1] ,dtype='float')
    max_fevals = 5*10**3
    d = 5  # bits in logic gate
    n_pop = 4+(3* (np.log(d))) #6.079
    cma_options = {'bounds':[cma_lb, cma_ub], 
                       'tolfun':1e-7, 
                       'maxfevals': max_fevals,
                       'verb_log': 0,
                       'verb_disp': 1, # print every iteration
                       'CMA_stds': np.abs(cma_ub - cma_lb)}
    
    #gates  = [gate1,gate2,gate3,gate4,gate5]
    savename = f"{gate}"
    
    f_CMA_1 = []
    x_CMA_1 = [] # list of solutions 
    #I = []
    while c <=30:
    #for i in range(30):
        
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
        init_sol = [x,y,z,t,u,v,p,w] 
        init_sigma = 0.5
        es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
        sol = es.optimize(neuro2lp).result 
        j = neuro2lp_gates(sol.xbest.tolist())
        if j < 10:
            c = c + 1
            x_CMA_1.append(sol.xbest.tolist())
            f_CMA_1.append(j) 
        print("--- %s seconds ---" % (time.time() - start_time))
        #print("--- %s minutes ---" % (time.time() - start_time)/60)
    # store X
    with open(f"Desktop/x_CMAES_{savename}.txt", "wb") as fp:   #Pickling
        pickle.dump(x_CMA_1, fp)   
    with open(f"Desktop/x_CMAES_{savename}.txt", "rb") as fp:   # Unpickling
        x_CMA_1 = pickle.load(fp)
    
    with open(f"Desktop/f_CMAES_{savename}.txt", "wb") as fp:   #Pickling
        pickle.dump(f_CMA_1 , fp)   
    with open(f"Desktop/f_CMAES_{savename}.txt", "rb") as fp:   # Unpickling
        f_CMA_1 = pickle.load(fp)
    


    