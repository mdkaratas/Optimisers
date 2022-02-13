#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:39:04 2021

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

def neuro1lp_gates(inputparams,gates):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
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
    sol = es.optimize(neuro1lp_gates,args = ([0,1],)).result 
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






#%%

#####################################################      CMA-ES: G x G  10


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


x_CMA_2 = []  # list of solutions 
for i in range(30):
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
    sol = es.optimize(neuro1lp_gates,args = ([1,0],)).result 
    x_CMA_2.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store X
with open("Desktop/x_CMA_2.txt", "wb") as fp:   #Pickling
    pickle.dump(x_CMA_2, fp)   
with open("Desktop/x_CMA_2.txt", "rb") as fp:   # Unpickling
    x_CMA_2 = pickle.load(fp)

f_CMA_2 = [] 
for i in x_CMA_2:
    f_CMA_2.append(neuro1lp_gates(i,[1,0]))   
                   
# store init X
with open("Desktop/f_CMA_2.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_2 , fp)   
with open("Desktop/f_CMA_2.txt", "rb") as fp:   # Unpickling
    f_CMA_2 = pickle.load(fp)
 
    
 #%%

 #####################################################      CMA-ES: G x G  11

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

x_CMA_3 = []  # list of solutions 
for i in range(30):
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
    sol = es.optimize(neuro1lp_gates,args = ([1,1],)).result 
    x_CMA_3.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store X
with open("Desktop/x_CMA_3.txt", "wb") as fp:   #Pickling
    pickle.dump(x_CMA_3, fp)   
with open("Desktop/x_CMA_3.txt", "rb") as fp:   # Unpickling
    x_CMA_3 = pickle.load(fp)

f_CMA_3 = [] 
for i in x_CMA_3 :
    f_CMA_3.append(neuro1lp_gates(i,[1,1]))   
                   
# store init X
with open("Desktop/f_CMA_3.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_3 , fp)   
with open("Desktop/f_CMA_3.txt", "rb") as fp:   # Unpickling
    f_CMA_3 = pickle.load(fp)
    
    
#%%

#####################################################      CMA-ES: G x G  00


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


x_CMA_0 = []  # list of solutions 
for i in range(30):
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
    sol = es.optimize(neuro1lp_gates,args = ([0,0],)).result 
    x_CMA_0.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store X
with open("Desktop/x_CMA_0.txt", "wb") as fp:   #Pickling
    pickle.dump(x_CMA_0, fp)   
with open("Desktop/x_CMA_0.txt", "rb") as fp:   # Unpickling
    x_CMA_0 = pickle.load(fp)

f_CMA_0 = [] 
for i in x_CMA_0 :
    f_CMA_0.append(neuro1lp_gates(i,[0,0]))   
                   
# store init X
with open("Desktop/f_CMA_0.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_0, fp)   
with open("Desktop/f_CMA_0.txt", "rb") as fp:   # Unpickling
    f_CMA_0 = pickle.load(fp)    
    


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

for i in init_X:
    if i[5:7] == [0.0, 1.0]:
        gate_1.append(i)
        f_g1.append(neuro1lp_costf(i))
    if i[5:7] == [1.0, 1.0] :
        gate_3.append(i)
        f_g3.append(neuro1lp_costf(i))
    if i[5:7] == [0.0, 0.0] :
        gate_0.append(i)
        f_g0.append(neuro1lp_costf(i))
    if i[5:7] == [1.0, 0.0] :
        gate_2.append(i)
        f_g2.append(neuro1lp_costf(i))


###  boxplot

data = [f_g1, f_g3, f_g0, f_g2]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '0',
                    '3', '2'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("NMMSO-MI")
plt.show()   
    