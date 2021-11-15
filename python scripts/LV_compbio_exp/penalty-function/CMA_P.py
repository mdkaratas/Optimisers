#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:57:49 2021

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

eng = matlab.engine.start_matlab()


#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost functions/neuro1lp_costfcn"
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

def neuro1lp_costf(inputparams):
    init = inputparams
    inputparams = inputparams[0:5]
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = init[5:8]
    gates = list(gates)
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost



# testing

def neuro1lp(inputparams):
    for i in inputparams:
        if i[0] + i [1] < 24:
           inputparams[5] = round(inputparams[5])
           inputparams[6] = round(inputparams[6])
           cost=neuro1lp_costf(inputparams)
        else:
            dist = i[0] + i [1] - 24
            cost = dist + neuro1lp_costf(inputparams)
    return cost



#####################################################################################   CMA-ES-MI + Penalty 


cma_lb = np.array([0,0,0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1,1,1] ,dtype='float')


max_fevals = 5*10**5
n_pop = 10
 
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}
                   #'popsize': n_pop}#'is_feasible': ff}

#inputparams = [12.0,6.1,3.5,0.4,0.6,0.04,0.7]

def neuro1lp(inputparams):
    for i in inputparams:
        if i[0] + i [1] < 24:
           inputparams[5] = round(inputparams[5])
           inputparams[6] = round(inputparams[6])
           cost=neuro1lp_costf(inputparams)
        else:
            dist = i[0] + i [1] - 24
            cost = dist + neuro1lp_costf(inputparams)
    return cost
#neuro1lp(inputparams)
 # I initial pts used
I = []
X = []  # list of solutions 
for i in range(1):
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    u = np.random.uniform(0,1)
    k = np.random.uniform(0,1)
    l = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,k,l] 
    I.append(init_sol)
    init_sigma = 0.5
    es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
    sol = es.optimize(neuro1lp).result 
    X.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store I
with open("Desktop/x_CMA_MI_P.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)   
with open("Desktop/x_CMA_MI_P.txt", "rb") as fp:   # Unpickling
    x_CMA_MI_P = pickle.load(fp)



F = [] 
for i in X:
    F.append(neuro1lp_costf(i))   
# store init F    
with open("Desktop/f_CMA_MI_P.txt", "wb") as fp:   #Pickling
    pickle.dump(F, fp)   
with open("Desktop/f_CMA_MI_P.txt", "rb") as fp:   # Unpickling
    f_CMA_MI_P = pickle.load(fp)         