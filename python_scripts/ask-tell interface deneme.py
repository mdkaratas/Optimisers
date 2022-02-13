#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:54:11 2021

@author: melikedila
"""

import numpy as np
import matlab.engine
import cma
import matplotlib.pyplot as plt
import random
import time
from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem
from pynmmso.listeners import TraceListener
import pickle
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

def neuro1lp(inputparams):
    for i in inputparams:
        inputparams = list(inputparams)
        inputparams[5] = round(inputparams[5])
        inputparams[6] = round(inputparams[6])
        cost=neuro1lp_costf(inputparams)     
    return cost


cma_lb = np.array([0,0,0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1,1,1] ,dtype='float')


max_fevals = 5*10**2
n_pop = 4+(3* (np.log(2))) #6.079
n_pop = 6
 
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}

# no need for lopp- look at the trajecory of CMA-ES
design = []
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
init_sigma = 0.5
es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
sol = es.optimize(neuro1lp).result
design.append(sol.xbest.tolist())
while not es.stop():
    fit, X = [], []
    while len(X) < es.popsize:
       curr_fit = None
       while curr_fit in (None, np.NaN):
             x = es.ask(1)[0]
             curr_fit = cma.ff.somenan(x, neuro1lp) # might return np.NaN
       X.append(x)
       fit.append(curr_fit)
    es.tell(X, fit)
    es.logger.add()
    es.disp()  #doctest: +ELLIPSIS


es.logger.plot() 
plt.savefig('Desktop/destination_path.eps', format='eps')
cma.s.figshow() 


X[0]



############################################################################### G x G NMMSO [0,1]

x_NMMSO_1 = []
f_NMMSO_1 = []

class MyProblem:
  @staticmethod
  def fitness(inputparams,gates=[0, 1]):
    inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    gates = matlab.double(gates)
    cost = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return - cost

  @staticmethod
  def get_bounds():
    return [0,0,0,0,0], [24,24,12,1,1]


trace_loc = []
for i in range(30):
    def main():
        number_of_fitness_evaluations = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem,swarm_size=10)
        nmmso.add_listener(TraceListener(level=2))
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_1.append(mode_result.location)
            f_NMMSO_1.append(mode_result.value)
        # The internals of the Nmmso object will be in the uniform parameter space
        for swarm in nmmso.swarms:
            trace_loc.append(problem.remap_parameters(swarm.mode_location))
            print("Swarm id: {} uniform parameter space location : {}  original parameter space location: {}".format(
                swarm.id, swarm.mode_location, problem.remap_parameters(swarm.mode_location)))


    if __name__ == "__main__":
        main()

with open("Desktop/x_NMMSO_1.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_1, fp)   
with open("Desktop/x_NMMSO_1.txt", "rb") as fp:   # Unpickling
    x_NMMSO_1 = pickle.load(fp)  


with open("Desktop/f_NMMSO_1.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_1, fp)   
with open("Desktop/f_NMMSO_1.txt", "rb") as fp:   # Unpickling
    f_NMMSO_1 = pickle.load(fp)  


#################################################   NMMSO 1-rum step through the iterations
trace_loc = []
trace_fitness = []
class MyProblem:
  @staticmethod
  def fitness(inputparams,gates=[0, 1]):
    inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    gates = matlab.double(gates)
    cost = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return - cost

  @staticmethod
  def get_bounds():
    return [0,0,0,0,0], [24,24,12,1,1]

def main():
    number_of_fitness_evaluations = 5*10**3
    #problem = UniformRangeProblem(MyProblem())    
    nmmso = Nmmso(MyProblem())

    while nmmso.evaluations < number_of_fitness_evaluations:
        nmmso.iterate()
        for swarm in nmmso.swarms:
            print("Swarm {} has {} particles.".format(swarm.id, swarm.number_of_particles))

    for mode_result in nmmso.get_result():
        trace_loc.append(mode_result.location)
        trace_fitness.append(mode_result.value)
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))


if __name__ == "__main__":
    main()






gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

x__NMMSO_MI = []
for i in x_NMMSO_MI:
    x__NMMSO_MI.append(i.tolist())
    
    
for i in x__NMMSO_MI:
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


###########################   line graph
plt.plot(xAxis,yAxis)
plt.title('title name')
plt.xlabel('xAxis name')
plt.ylabel('yAxis name')
plt.show()


