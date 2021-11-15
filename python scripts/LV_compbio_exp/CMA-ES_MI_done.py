
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:26:51 2021

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

###########################################################   CMA-ES : MI


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
with open("Desktop/x_CMA_MI.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)   
with open("Desktop/x_CMA_MI.txt", "rb") as fp:   # Unpickling
    x_CMA_MI = pickle.load(fp)



F = [] 
for i in X:
    F.append(neuro1lp_costf(i))   
# store init F    
with open("Desktop/f_CMA_MI.txt", "wb") as fp:   #Pickling
    pickle.dump(F, fp)   
with open("Desktop/f_CMA_MI.txt", "rb") as fp:   # Unpickling
    f_CMA_MI = pickle.load(fp)      
    


###########################################  
#%%

es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) 

g_X, g_f = [], []  
for i in range(30):
    start_time = time.time()
    while not es.stop(): 
        fit, X = [], []
        while len(X) < es.popsize:
            curr_fit = None
            while curr_fit in (None, np.NaN):
                x = es.ask(1)[0]
                curr_fit = neuro1lp(x)  # might return np.NaN
            X[i].append(x)
            fit[i].append(curr_fit)
        g_X[i].append(X)
        g_f[i].append(fit)    
        es.tell(X, fit)
        #es.logger.add()
        es.disp()
    print("--- %s seconds ---" % (time.time() - start_time))

#%% 
    
    

gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

for i in x_CMA_MI:
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

x = ['1','2','3','0']
y = [len(gate_1),len(gate_2), len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,15))
plt.yticks(np.arange(min(y), max(y)+1, 1.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI')
plt.savefig('Desktop/CMA-ES: MI.eps', format='eps')
 



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


