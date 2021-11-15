#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:13:59 2021

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
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro1lp_costfcn"
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

#%%  generate initial solution and specify gates


x = random.uniform(0,24)
y = random.uniform(0,24)
while x+y > 24 :
    x = random.uniform(0,24)
    y = random.uniform(0,24)
z = np.random.uniform(0,12) 
t = np.random.uniform(0,1) 
u = np.random.uniform(0,1)
#k = np.random.uniform(0,1)
#l = np.random.uniform(0,1)
init_sol = [x,y,z,t,u] 
init_sol = matlab.double([init_sol])

gates=[0,1]
gates = matlab.double([gates])

#%%  call cost function

res=eng.getBoolCost_neuro1lp(init_sol,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=5)
cost = res[0]
sol_ld = res[1]
sol_ld['x']
sol_ld['y']
sol_dd = res[2]  
sol_dd['x']
sol_dd['y']
dat_ld = res[3]  
dat_ld['x']
dat_ld['y']
dat_dd = res[4]  
dat_dd['x']
dat_dd['y']
#%%

eng.bdeplot_hmaps(sol_ld)  # plots multiple Boolean timeseries as black and white heatmaps

eng.plot_neuro1lp_solns(dat_ld, sol_ld)  # plots comparisons between the thresholded data and corresponding prediction for the Boolean 1-loop Neurospora model, as heatmaps

eng.plot_neuro1lp_solns(eng.struct(dat_ld), eng.struct(sol_ld))

plot_neuro1lp_solns(dat_ld,sol_ld)


dat_ld = eng.struct(dat_ld)

sol_ld = eng.struct(sol_ld)


import PlotDrawer
myPlotter = PlotDrawer.initialize()



