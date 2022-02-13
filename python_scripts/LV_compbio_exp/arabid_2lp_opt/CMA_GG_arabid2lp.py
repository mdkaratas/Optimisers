#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:09:16 2022

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
n = 8
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

#####################################################      CMA-ES: G x G
for k in range(152,153):
    gate = gatesm[k]
    c = 0
    def arabid2lp_gates(inputparams):
        inputparams = list(inputparams)
        inputparams = matlab.double([inputparams])
        gates = gate
        gates = matlab.double([gates])
        cost=eng.getBoolCost_arabid2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        return cost
    
    def arabid2lp(inputparams):
        for i in inputparams:
            dist = []
            if (inputparams[0] + inputparams[1] + inputparams[2] >= 24) :
                dist.append(inputparams[0] + inputparams[1] + inputparams[2] - 23.5)
                
            if inputparams[4] + inputparams[5] >= 24:
                dist.append(inputparams[4] + inputparams[5] - 23.5)
                
            if inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] >= 24:
                dist.append(inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] - 23.5)  
            
            distance = sum(dist)
            cost = arabid2lp_gates(inputparams) + distance
                
        return cost     #inputparams = [12.0,6.1,3.5,0.4,0.6,0.04,0.7]
    
    cma_lb = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype='float')
    cma_ub = np.array([24,24,24,24,24,24,12,12,12,1,1,1,1,4,4] ,dtype='float')
    max_fevals = 5*10**3
    d = 8  # bits in logic gate
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
        for i in range(1,7):
            globals()['x_%s' % i]  = random.uniform(0,24)
        for i in range(7,10):
            globals()['x_%s' % i]  = random.uniform(0,12)    
        for i in range(10,14):
            globals()['x_%s' % i]  = random.uniform(0,1)    
        for i in range(14,15):
            globals()['x_%s' % i]  = random.uniform(0,4)  
        while any([ x_1+ x_2 +x_3 >= 24, x_5 + x_6 >= 24, x_2 + x_3 + x_4 + x_6 >= 24 ]):
            for i in range(1,7):
                globals()['x_%s' % i]  = random.uniform(0,24)
            for i in range(7,10):
                globals()['x_%s' % i]  = random.uniform(0,12)    
            for i in range(10,14):
                globals()['x_%s' % i]  = random.uniform(0,1)    
            for i in range(14,15):
                globals()['x_%s' % i]  = random.uniform(0,4)  

        init_sol = [globals()['x_%s' % i] for i in range(1,16) ] 
        #I.append(init_sol)
        init_sigma = 0.5
        es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
        sol = es.optimize(arabid2lp).result 
        j = neuro2lp_gates(sol.xbest.tolist())
        if j < 8:
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
        
        



















        
        
        
        