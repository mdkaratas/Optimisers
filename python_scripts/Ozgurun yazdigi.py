#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:19:15 2021

"""

#%%

# Import required packages

import matlab.engine
import numpy as np
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

# Define the input parameters

test_delays = [4.46, 6.27, 10.43]
test_thresholds=[0.38, 0.59]
test_inputparams = test_delays + test_thresholds
gates= [0,1]

#%%

# Define the cost function - list inputs
#=[0,1]
def neuro1lp_costf(inputparams,gates):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost
#%%
gates = [0,1]    
neuro1lp_costf([4.46, 6.27, 10.43,0.38, 0.59],gates)
neuro1lp_costf([5.06498114, 5.68913637, 9.85148075, 0.38081232, 0.58703905],[0,1])
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


def neuro1lp_costf(inputparams):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

def neuro1lp_costf(inputparams):
    gates = [0,1]
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    a=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    gates = [1,0]
    gates = matlab.double([gates])
    b= eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    gates = [0,0]
    gates = matlab.double([gates])
    c = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    gates = [1,1]
    gates = matlab.double([gates])
    d = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    cost =[a,b,c,d]  
    return cost
neuro1lp_costf([4.46,6.27, 10.43,0.38, 0.59])
import cma
import numpy as np

#def camel_back(x):
    #return ( ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2))
#gates = [0,1]
cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1] ,dtype='float')


max_fevals = 5*10**2
n_pop = 4+(3* (np.log(2))) #6.079
n_pop = 6
 
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}
                   #'popsize': n_pop}#'is_feasible': ff}

import random
x = random.uniform(0,24) 
y= random.uniform(0,24)  
z = random.uniform(0,12) 
t =random.uniform(0,1) 
u =random.uniform(0,1)      
init_sol = [x,y,z,t,u] 
#[4.46,6.27,10.43,0.38,0.59]
init_sigma = 0.5
es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
sol = es.optimize(neuro1lp_costf).result 
sol = es.optimize(neuro1lp_costf,args = ([1,1],)).result 
sol.xbest
a = sol.xbest.tolist()
neuro1lp_costf(a)
sol.fbest

import matplotlib.pyplot as plt
data_1 = [0.124031008,0.394379845,0.0881782945736434,0.087209302,0.114825581,0.316860465,0.21124031,0.088178295
,0.117005814 ,0.126453488 ,0.296027132, 0.082364341 ,0.399224806,0.073158915,0.067344961,0.167151163,0.126453488]

data_2 = [3.114583333,2.399224806,2.459060078,3.03125,3.057291667,2.261627907,3.057291667,3.041666667,
3.109375,3.078125,3.046875,3.0625,2.392926357,3.041666667,3.041666667,3.057291667,3.0625]

data_3 = [4,3.348837209,4,4,4,4,3.447674419,4,4,4,4,4,4,4,4,4,4]
data_4 = [4,4,4,4,4, 3.3779069767441863,4,4,4,4,4,4,4,4,4,4,4]


data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
ax.set_xticklabels(['2', '1',
                    '3', '0'])
plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES")
plt.show()

while not es.stop():
    X = es.ask()
    es.tell(X, [cma.fcts.neuro1lp_costf(x) for x in X])
    es.disp()


#neuro1lp_costf([5.00138738, 5.75336812, 9.95072781, 0.34573547, 0.58478232],[0,1])

es.result_pretty()

logger = cma.CMADataLogger().register(sol)
sol.xbest
sol.fbest
sol.evals_best
sol.evaluations
sol.xfavorite # distribution mean in "phenotype" space, to be considered as current best estimate of the optimum
sol.stds
sol.stop
es.logger.plot() 

while not es.stop():
    es.tell(*es.ask_and_eval(neuro1lp_costf))



import cma; cma.plot()
cma.plot();
cma.show()
cma.CMADataLogger().plot()

cma.disp()
logger = cma.CMADataLogger()
logger.plot()

neuro1lp_costf([0.93523505, 9.76780196, 0.02336601, 0.2691402 , 0.59345947],[0,1])

from matplotlib import pyplot
pyplot.show()





es = cma.fmin2(neuro1lp_costf, init_sol, 1,
    {'CMA_diagonal':True,
    # 'CMA_mirrormethod': 0,
    'CMA_mirrors':True,
    # 'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaCSA,
    'popsize': 160,
    'maxiter': 1200,
    'ftarget': 1e-9,
})[1]
cma.plot()
input()

############################  NMMSO
#for maximisation normally 

class MyProblem:
  @staticmethod
  def fitness(inputparams,gates=[0, 0]):
    inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    gates = matlab.double(gates)
    cost = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return - cost

  @staticmethod
  def get_bounds():
    return [0,0,0,0,0], [24,24,12,1,1]



from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem

def main():
    number_of_fitness_evaluations = 5e+4
    problem = UniformRangeProblem(MyProblem())

    nmmso = Nmmso(problem,swarm_size=10)
    
    while nmmso.evaluations < number_of_fitness_evaluations:
        nmmso.iterate()
        for swarm in nmmso.swarms:
            print("Swarm {} has {} particles.".format(swarm.id, swarm.number_of_particles))
    
    for mode_result in nmmso.get_result():
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))


if __name__ == "__main__":
    main()





neuro1lp_costf([5.50120035 ,5.29908878 ,9.79104634 ,0.3696 , 0.60085809],[0,1])




number_of_fitness_evaluations = 5e+4
problem = UniformRangeProblem(MyProblem())

nmmso = Nmmso(problem,swarm_size=10)

while nmmso.evaluations < number_of_fitness_evaluations:
    nmmso.iterate()
    for swarm in nmmso.swarms:
        print("Swarm {} has {} particles.".format(swarm.id, swarm.number_of_particles))

    for mode_result in nmmso.get_result():
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))





def main():
    number_of_fitness_evaluations = 5000
    nmmso = Nmmso(MyProblem,swarm_size=10)
    my_result = nmmso.run(number_of_fitness_evaluations)
    for mode_result in my_result:
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))


if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt

#matplotlib.pyplot.boxplot(data)


data = np.random.normal(100, 20, 200)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(data)
plt.show()





import cma
opts = cma.CMAOptions()
opts.set("bounds", [[-2, None], [2, None]])
cma.fmin(cost_function, x_start, sigma_start, opts)






data_1 = [0.124031008,0.394379845,0.0881782945736434,0.087209302,0.114825581,0.316860465,0.21124031,0.088178295
,0.117005814 ,0.126453488 ,0.296027132, 0.082364341 ,0.399224806,0.073158915,0.067344961,0.167151163,0.126453488]

data_2 = [2.536458333,2.740915698,2.5625,0.867974806,1.34193314,2.154554264,1.714026163,2.885416667,2.583333333,
2.484375,2.661458333,3.234375,1.565164729,2.479166667,1.416182171,2.541666667]

data_3 = [4,3.348837209,4,4,4,4,3.447674419,4,4,4,4,4,4,4,4,4,4]
data_4 = [4,4,4,4,4, 3.3779069767441863,4,4,4,4,4,4,4,4,4,4,4]


data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
ax.set_xticklabels(['2', '1',
                    '3', '0'])
plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("NMMSO")
plt.show()
