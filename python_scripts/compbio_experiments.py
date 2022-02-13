#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:24:38 2021

@author: melikedila
"""

#%%  
#   install required packages
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

# cost

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


#%%
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


for i in range(30):
    start_time = time.time()
    def main():
        number_of_fitness_evaluations = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        #nmmso.add_listener(TraceListener(level=2))
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_1.append(mode_result.location)
            f_NMMSO_1.append(mode_result.value)
        # The internals of the Nmmso object will be in the uniform parameter space
        for swarm in nmmso.swarms:
            print("Swarm id: {} uniform parameter space location : {}  original parameter space location: {}".format(
                swarm.id, swarm.mode_location, problem.remap_parameters(swarm.mode_location)))

    if __name__ == "__main__":
        main()
    print("--- %s seconds ---" % (time.time() - start_time))
        
with open("Desktop/x_NMMSO_1.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_1, fp)   
with open("Desktop/x_NMMSO_1.txt", "rb") as fp:   # Unpickling
    x_NMMSO_1 = pickle.load(fp)  


with open("Desktop/f_NMMSO_1.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_1, fp)   
with open("Desktop/f_NMMSO_1.txt", "rb") as fp:   # Unpickling
    f_NMMSO_1 = pickle.load(fp)  

#%%
############################################################################### G x G NMMSO [1,0]

x_NMMSO_2 = []
f_NMMSO_2 = []

class MyProblem:
  @staticmethod
  def fitness(inputparams,gates=[1, 0]):
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
    start_time = time.time()
    def main():
        number_of_fitness_evaluations = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        #nmmso.add_listener(TraceListener(level=2))
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_2.append(mode_result.location)
            f_NMMSO_2.append(mode_result.value)
        # The internals of the Nmmso object will be in the uniform parameter space
        for swarm in nmmso.swarms:
            trace_loc.append(problem.remap_parameters(swarm.mode_location))
            print("Swarm id: {} uniform parameter space location : {}  original parameter space location: {}".format(
                swarm.id, swarm.mode_location, problem.remap_parameters(swarm.mode_location)))


    if __name__ == "__main__":
        main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
with open("Desktop/x_NMMSO_2.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_2, fp) 
    
with open("Desktop/x_NMMSO_2.txt", "rb") as fp:   # Unpickling
    x_NMMSO_2 = pickle.load(fp)  


with open("Desktop/f_NMMSO_2.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_2, fp)   
with open("Desktop/f_NMMSO_2.txt", "rb") as fp:   # Unpickling
    f_NMMSO_2 = pickle.load(fp)  


#%%
############################################################################### G x G NMMSO [1,1]

x_NMMSO_3 = []
f_NMMSO_3 = []

class MyProblem:
  @staticmethod
  def fitness(inputparams,gates=[1, 1]):
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
    start_time = time.time()
    def main():
        number_of_fitness_evaluations = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_3.append(mode_result.location)
            f_NMMSO_3.append(mode_result.value)

    if __name__ == "__main__":
        main()
    print("--- %s seconds ---" % (time.time() - start_time))
with open("Desktop/x_NMMSO_3.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_3, fp)   
with open("Desktop/x_NMMSO_3.txt", "rb") as fp:   # Unpickling
    x_NMMSO_3 = pickle.load(fp)  


with open("Desktop/f_NMMSO_3.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_3, fp)   
with open("Desktop/f_NMMSO_3.txt", "rb") as fp:   # Unpickling
    f_NMMSO_3 = pickle.load(fp)  


#%%
############################################################################### G x G NMMSO [0,0]

x_NMMSO_0 = []
f_NMMSO_0 = []

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


trace_loc = []
for i in range(30):
    start_time = time.time()
    def main():
        number_of_fitness_evaluations = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        #nmmso.add_listener(TraceListener(level=2))
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_0.append(mode_result.location)
            f_NMMSO_0.append(mode_result.value)
        # The internals of the Nmmso object will be in the uniform parameter space
        #for swarm in nmmso.swarms:
            #trace_loc.append(problem.remap_parameters(swarm.mode_location))
            #print("Swarm id: {} uniform parameter space location : {}  original parameter space location: {}".format(
                #swarm.id, swarm.mode_location, problem.remap_parameters(swarm.mode_location)))


    if __name__ == "__main__":
        main()
    print("--- %s seconds ---" % (time.time() - start_time))
with open("Desktop/x_NMMSO_0.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_0, fp)   
with open("Desktop/x_NMMSO_0.txt", "rb") as fp:   # Unpickling
    x_NMMSO_0 = pickle.load(fp)  


with open("Desktop/f_NMMSO_0.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_0, fp)   
with open("Desktop/f_NMMSO_0.txt", "rb") as fp:   # Unpickling
    f_NMMSO_0 = pickle.load(fp)  
    
 #%%   
    
###################################################   NMMSO : MI

f_NMMSO_MI = []
x_NMMSO_MI = []


class MyProblem:
  @staticmethod
  def fitness(inputparams):
    for i in inputparams:
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
       cost=neuro1lp_costf(inputparams)     
       return - cost
    
  @staticmethod
  def get_bounds():
    return [0,0,0,0,0,0,0], [24,24,12,1,1,1,1]
    
    

for i in range(30):
    start_time = time.time()   
    def main():
        number_of_fitness_evaluations = 5*10**4
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        #nmmso.add_listener(TraceListener(level=2))
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            #print("Mode at {} has value {}".format(mode_result.location, mode_result.value))
            x_NMMSO_MI.append(mode_result.location)
            f_NMMSO_MI.append(mode_result.value)
        # The internals of the Nmmso object will be in the uniform parameter space
        #for swarm in nmmso.swarms:
            #trace_loc.append(problem.remap_parameters(swarm.mode_location))
            #print("Swarm id: {} uniform parameter space location : {}  original parameter space location: {}".format(
                #swarm.id, swarm.mode_location, problem.remap_parameters(swarm.mode_location)))


    if __name__ == "__main__":
        main()
    print("--- %s seconds ---" % (time.time() - start_time))
        

    
 
    
with open("Desktop/x_NMMSO_MI.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_MI, fp)   
with open("Desktop/x_NMMSO_MI.txt", "rb") as fp:   # Unpickling
    x_NMMSO_MI = pickle.load(fp)  


with open("Desktop/f_NMMSO_MI.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_MI, fp)   
with open("Desktop/f_NMMSO_MI.txt", "rb") as fp:   # Unpickling
    f_NMMSO_MI= pickle.load(fp)  

    
for i in range(len(x_NMMSO_MI)):
    x_NMMSO_MI[i][5] = round(x_NMMSO_MI[i][5])
    x_NMMSO_MI[i][6] = round(x_NMMSO_MI[i][6])

##########################  plots NMMSO- MI

x_NMMSO_MI = [ list(i) for i in x_NMMSO_MI ]
for i in x_NMMSO_MI:
    print( i[5:7])

x_NMMSO_MI[0]


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

for i in x_NMMSO_MI:
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


#  Bar chart- MI gates 

x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,15))
plt.yticks(np.arange(min(y), max(y)+1, 1.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('NMMSO: MI')
 



##  PCP MI parameter search

gate_0x = []
for i in gate_0:
    gate_0x.append(i[0:5])


#from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
cmap = plt.get_cmap('jet')
t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'


x = [r'${}$'.format(t) ,r'${}$'.format(e), r'${}$'.format(p) , r'${}$'.format(c) , r'${}$'.format(s)]
fig,(ax,ax2,ax3,ax4) = plt.subplots(1,4,sharey = False,dpi= 100)
plt.rcParams['figure.figsize'] = [3,6]
plt.rcParams["axes.prop_cycle"] = plt.cycler("color",cmap(np.linspace(0,1,2)))

a = ax.plot(x,gate_0x[0],x,gate_0x[1])
b = ax2.plot(x,gate_0x[0],x,gate_0x[1])
c = ax3.plot(x,gate_0x[0],x,gate_0x[1])
d = ax4.plot(x,gate_0x[0],x,gate_0x[1])
#e = ax5.plot(x,gate_0x[0],x,gate_0x[1])

fig.text(0.5,0.06,'Parameters', ha = 'center')
fig.text(0.0,0.5,'Parameter value',va = 'center',rotation = 'vertical')

# zoom in each of the subplots
ax.set_xlim(x[0],x[1])
ax2.set_xlim(x[1],x[2])
ax3.set_xlim(x[2],x[3])
ax4.set_xlim(x[3],x[4])

ax.tick_params(axis="y", labelsize=6)
ax2.tick_params(axis="y", labelsize=6)
ax3.tick_params(axis="y", labelsize=6)
ax4.tick_params(axis="y", labelsize=6)
plt.tick_params(axis='y', which='both', labelleft=True, labelright=True)
#ax5 = ax4.twinx()
#ax5.set_xlim(x[4],x[5])
# stack the subplots together
plt.subplots_adjust(wspace=0)
#cax = fig.add_axes([0.99, 0.2, 0.2, 0.32])
#fig.colorbar(cm.ScalarMappable(cmap=cmap),orientation='vertical',cax=cax)
plt.colorbar(cm.ScalarMappable(cmap=cmap),pad=0.1)
plt.show()



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



#%%

###########################################################   CMA-ES : MI


cma_lb = np.array([0,0,0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1,1,1] ,dtype='float')


max_fevals = 5*10**3
n_pop = 4+(3* (np.log(2))) #6.079
n_pop = 6
 
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
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
       cost=neuro1lp_costf(inputparams)     
    return cost
#neuro1lp(inputparams)
 # I initial pts used
I = []
X = []  # list of solutions 
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
with open("Desktop/init_pts.txt", "wb") as fp:   #Pickling
    pickle.dump(I, fp)   
with open("Desktop/init_pts.txt", "rb") as fp:   # Unpickling
    prev_pts = pickle.load(fp)

# store init X
with open("Desktop/init_X.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)   
with open("Desktop/init_X.txt", "rb") as fp:   # Unpickling
    init_X = pickle.load(fp)

F = [] 
for i in X:
    F.append(neuro1lp_costf(i))   
# store init F    
with open("Desktop/init_F.txt", "wb") as fp:   #Pickling
    pickle.dump(F, fp)   
with open("Desktop/init_F.txt", "rb") as fp:   # Unpickling
    init_F = pickle.load(fp)      
    

x_CMA_MI = init_X
f_CMA_MI = init_F
with open("Desktop/X_CMA_MI.txt", "wb") as fp:   #Pickling
    pickle.dump(x_CMA_MI, fp)  
with open("Desktop/F_CMA_MI.txt", "rb") as fp:   #Pickling
    pickle.dump(f_CMA_MI, fp)      
    
    

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


#  Bar chart- MI gates 

x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,15))
plt.yticks(np.arange(min(y), max(y)+1, 1.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI')
 



##  PCP MI parameter search

gate_0x = []
for i in gate_0:
    gate_0x.append(i[0:5])


#from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
cmap = plt.get_cmap('jet')
t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'


x = [r'${}$'.format(t) ,r'${}$'.format(e), r'${}$'.format(p) , r'${}$'.format(c) , r'${}$'.format(s)]
fig,(ax,ax2,ax3,ax4) = plt.subplots(1,4,sharey = False,dpi= 100)
plt.rcParams['figure.figsize'] = [3,6]
plt.rcParams["axes.prop_cycle"] = plt.cycler("color",cmap(np.linspace(0,1,2)))

a = ax.plot(x,gate_0x[0],x,gate_0x[1])
b = ax2.plot(x,gate_0x[0],x,gate_0x[1])
c = ax3.plot(x,gate_0x[0],x,gate_0x[1])
d = ax4.plot(x,gate_0x[0],x,gate_0x[1])
#e = ax5.plot(x,gate_0x[0],x,gate_0x[1])

fig.text(0.5,0.06,'Parameters', ha = 'center')
fig.text(0.0,0.5,'Parameter value',va = 'center',rotation = 'vertical')

# zoom in each of the subplots
ax.set_xlim(x[0],x[1])
ax2.set_xlim(x[1],x[2])
ax3.set_xlim(x[2],x[3])
ax4.set_xlim(x[3],x[4])

ax.tick_params(axis="y", labelsize=6)
ax2.tick_params(axis="y", labelsize=6)
ax3.tick_params(axis="y", labelsize=6)
ax4.tick_params(axis="y", labelsize=6)
plt.tick_params(axis='y', which='both', labelleft=True, labelright=True)
#ax5 = ax4.twinx()
#ax5.set_xlim(x[4],x[5])
# stack the subplots together
plt.subplots_adjust(wspace=0)
#cax = fig.add_axes([0.99, 0.2, 0.2, 0.32])
#fig.colorbar(cm.ScalarMappable(cmap=cmap),orientation='vertical',cax=cax)
plt.colorbar(cm.ScalarMappable(cmap=cmap),pad=0.1)
plt.show()



###  boxplot

data = [f_g1, f_g3, f_g0, f_g2]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '0',
                    '3', '2'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:MI")
plt.show()


#%%

#####################################################      CMA-ES: G x G  01

def neuro1lp_gates(inputparams,gates):
    for i in inputparams:
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1] ,dtype='float')
max_fevals = 5*10**3
n_pop = 4+(3* (np.log(2))) #6.079
n_pop = 6
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb)}

 # I initial pts used
x_CMA_1 = [] # list of solutions 
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
    k = np.random.uniform(0,1)
    l = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,k,l] 
    I.append(init_sol)
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
for i in X:
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
max_fevals = 5*10**2
n_pop = 4+(3* (np.log(2))) #6.079
n_pop = 6
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
    k = np.random.uniform(0,1)
    l = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,k,l] 
    I.append(init_sol)
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
for i in X:
    f_CMA_2.append(neuro1lp_gates(i,[1,0]))   
                   
# store init X
with open("Desktop/f_CMA_2.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_2 , fp)   
with open("Desktop/f_CMA_2.txt", "rb") as fp:   # Unpickling
    f_CMA_2 = pickle.load(fp)
 
    
 #%%

 #####################################################      CMA-ES: G x G  11
def neuro1lp_gates(inputparams,gates):
    for i in inputparams:
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1] ,dtype='float')
max_fevals = 5*10**3

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
    k = np.random.uniform(0,1)
    l = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,k,l] 
    I.append(init_sol)
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
for i in X:
    f_CMA_3.append(neuro1lp_gates(i,[1,1]))   
                   
# store init X
with open("Desktop/f_CMA_3.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_3 , fp)   
with open("Desktop/f_CMA_3.txt", "rb") as fp:   # Unpickling
    f_CMA_3 = pickle.load(fp)
    
    
#%%

#####################################################      CMA-ES: G x G  00
def neuro1lp_gates(inputparams,gates):
    for i in inputparams:
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

cma_lb = np.array([0,0,0,0,0], dtype='float')
cma_ub = np.array([24,24,12,1,1] ,dtype='float')
max_fevals = 5*10**3
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
    k = np.random.uniform(0,1)
    l = np.random.uniform(0,1)
    init_sol = [x,y,z,t,u,k,l] 
    I.append(init_sol)
    init_sigma = 0.5
    es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 
    sol = es.optimize(neuro1lp_gates,args = ([0,0],)).result 
    X.append(sol.xbest.tolist())
    print("--- %s seconds ---" % (time.time() - start_time))
    #print("--- %s minutes ---" % (time.time() - start_time)/60)
# store X
with open("Desktop/x_CMA_0.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)   
with open("Desktop/x_CMA_0.txt", "rb") as fp:   # Unpickling
    x_CMA_0 = pickle.load(fp)

f_CMA_0 = [] 
for i in X:
    f_CMA_0.append(neuro1lp_gates(i,[0,0]))   
                   
# store init X
with open("Desktop/f_CMA_0.txt", "wb") as fp:   #Pickling
    pickle.dump(f_CMA_0, fp)   
with open("Desktop/f_CMA_0.txt", "rb") as fp:   # Unpickling
    f_CMA_0 = pickle.load(fp)    
    
    
    
    
    
    
    
    
    