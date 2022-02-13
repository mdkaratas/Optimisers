#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:01:19 2022

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
import itertools


eng = matlab.engine.start_matlab()


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
root = '/Users/melikedila/Documents/GitHub/'

path= root + r"/BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= path= root + r"/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/models"
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



#############################################  CMA MI

read_root = 'Desktop/Llyonesse/Arabid_2lp_res/'

with open(read_root + "x_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   
    x_CMA_MI = pickle.load(fp)   
with open(read_root + "f_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   
    f_CMA_MI = pickle.load(fp)  

#############################################  NMMSO MI

with open(read_root + "design_dict_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)   
with open(read_root + "fit_dict_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)   
with open(read_root + "f_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    f_NMMSO_MI = pickle.load(fp)  
with open(read_root + "x_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)       
   
#############################################  CMA-ES GG

n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(len(gatesm)):
    gate = gatesm[k]
    savename = f"{gate}"    
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/x_CMAES_{savename}_cts.txt", "rb") as fp:   
        globals()['x_CMAES_%s' % gate] = pickle.load(fp) 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/f_CMAES_{savename}_cts.txt", "rb") as fp:   
        globals()['f_CMAES_%s' % gate]= pickle.load(fp) 
    globals()['x_CMAES_%s' % gate] = globals()['x_CMAES_%s' % gate][0:30]
    globals()['f_CMAES_%s' % gate] = globals()['f_CMAES_%s' % gate][0:30]
  
#############################################  NMMSO GG


for k in range(len(gatesm)):
    gate = gatesm[k]
    savename = f"{gate}" 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/design_dict_{savename}_NMMSO_cts.txt", "rb") as fp:   
        globals()['design_dict_NMMSO_%s' % gate] = pickle.load(fp)   
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/fit_dict_{savename}_NMMSO_cts.txt", "rb") as fp:   
        globals()['fit_dict_NMMSO_%s' % gate] = pickle.load(fp)     
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/x_NMMSO_{savename}_NMMSO_cts.txt", "rb") as fp:   
        globals()['x_NMMSO_%s' % gate] = pickle.load(fp) 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/f_NMMSO_{savename}_NMMSO_cts.txt", "rb") as fp:   
        globals()['f_NMMSO_%s' % gate]= pickle.load(fp) 


##################  PLOT.

for k in range(256):
    gate = gatesm[k]
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

a =[11.0, 6.0, 2.0, 1.0, 11.0, 2.0, 10.0, 5.0, 1.5861735045618337, 0.8212703758320038, 0.13895538388844852, 0.9904224770196597, 0.8685118175928863, 0.0018412457802242245, 3.984895340248584, 0.776964366397019, 0.005963156027925716, 0.0022028810948015844, 0.9998369257303865, 0.26059354845544813, 0.4040223744744524, 0.259160774003397, 0.35049207585479064]
for i in range(8):
    inputparams[i] = round(inputparams[i]) 
    for j,i in enumerate(x_CMA_MI):
        if i[15:24] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(f_CMA_MI[j])
            
x  = []  
y = []   
for k in range(256): 
   x.append(str(k))  
   y.append(len(globals()['gate_%s' % k]))
            
opt = np.column_stack((x, y))
opt = opt[np.argsort(opt[:, 1])]
x = opt[:,0]
y = list(opt[:,1])

X = []
for i in x:
    X.append(str(i))   
Y = []
for i in y:
    Y.append(int(i))     
y = Y

x_list,y_list = [], []
for i in range(32):
    x_list.append(x[31-i])
    y_list.append(y[31-i])
    
x_list = x_list[0:11]
y_list = y_list[0:11]

fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=12,rotation=90)
bar = plt.bar(x_list, height=y_list,color= 'royalblue')
bar[1].set_color('purple')
plt.title('CMA-ES: MI',fontsize=25)
plt.savefig('Desktop/cont_figs/CMA_ES_MI_frequency_2lp.eps', format='eps',bbox_inches='tight')






























