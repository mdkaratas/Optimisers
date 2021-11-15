#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:19:15 2021

@author: oea201
"""

#%%

# Import required packages

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
eng = matlab.engine.start_matlab()

#%%

# Set required paths

path= r"/Users/melikedila/Documents/Repos/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/models"
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

# Define the input parameters

test_delays = [5.0752, 6.0211, 14.5586]
test_thresholds=[0.3, 0.3]
test_inputparams = test_delays + test_thresholds


#%%

# Define the cost function - list inputs

def neuro1lp_costf(inputparams,gates=[0, 1]):
    test_inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    gates = matlab.double(gates)
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost
    

#%%

# Define the cost function - nparray inputs

def fneuro1lp_costf(inputparams,gates=np.array([0, 1])):
    inputparams = matlab.double(inputparams.tolist())
    gates = matlab.double(gates.tolist())
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost


#%%

# Define the cost function - nparray inputs with only parameters as input.

gates = np.array([0, 1])
gates = matlab.double(gates.tolist())

def fneuro1lp_costf2(inputparams):
    inputparams = matlab.double(inputparams.tolist())
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

#%%

# Try and reproduce known combinations - list inputs

test_gates = [0, 1]
test_cost = neuro1lp_costf(test_inputparams,test_gates)

#%%

# Try and reproduce known combinations - nparray inputs

ftest_inputparams = np.array(test_inputparams)
ftest_gates= np.array(test_gates)
ftest_cost = fneuro1lp_costf(ftest_inputparams,ftest_gates)

#%%

# Try and reproduce known combinations - nparray inputs with fixed gates

ftest_inputparams = np.array(test_inputparams)
ftest_cost2 = fneuro1lp_costf2(ftest_inputparams)



import cma
import numpy as np

def camel_back(x):
    return ( ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2))

cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,24,1,1] ,dtype='float')

max_fevals = 5*10**4
n_pop = 100
 
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb),
                   'popsize': n_pop}#'is_feasible': ff}
init_sol = test_inputparams
init_sigma = 0.5
es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options ) # optimizes the 2-dimensional function with initial solution all zeros and initial sigma = 0.5
es.result_pretty()
sol = es.optimize(neuro1lp_costf,args = ([0,1],)).result   

sol.xbest
sol.fbest
sol.evals_best
sol.evaluations
sol.xfavorite # distribution mean in "phenotype" space, to be considered as current best estimate of the optimum
sol.stds
sol.stop

import cmaplt
cma.plot();
cma.show()
cma.CMADataLogger().plot()

cma.disp()
logger = cma.CMADataLogger()
logger.plot()

neuro1lp_costf([0.93523505, 9.76780196, 0.02336601, 0.2691402 , 0.59345947],[0,1])

