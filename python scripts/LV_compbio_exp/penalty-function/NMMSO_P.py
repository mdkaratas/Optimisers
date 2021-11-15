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

#####################################################################################   NMMSO-MI + Penalty
#%%  

f_NMMSO_MI = []
x_NMMSO_MI = []


    
class MyProblem:
  @staticmethod
  def fitness(inputparams):
    for i in inputparams:
       inputparams[5] = round(inputparams[5])
       inputparams[6] = round(inputparams[6])
       cost=neuro1lp(inputparams)     
       return - cost
    
  @staticmethod
  def get_bounds():
    return [0,0,0,0,0,0,0], [24,24,12,1,1,1,1]
    

design_dict = {}
fit_dict = {}
for i in range(30):
    start_time = time.time()
    local_modes =[]
    local_fit = []
    def main():
        number_of_fitness_evaluations = 5*10**4
        problem = UniformRangeProblem(MyProblem())
        nmmso = Nmmso(problem,swarm_size=10)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            local_modes.append(mode_result.location)
            local_fit.append(mode_result.value)
        design_dict[i] = local_modes
        fit_dict[i] = local_fit
        print("--- %s seconds ---" % (time.time() - start_time))   
    if __name__ == "__main__":
        main()


x_NMMSO__MI = []
f_NMMSO_MI = []
k = 0
for key,value in fit_dict.items():
    f = max(value)
    f_NMMSO_MI.append(f)
    id = value.index(f)
    l =  list(design_dict[key][id])
    x_NMMSO_MI.append(l)


with open("design_dict_NMMSO_MI_P.txt", "wb") as fp:   
    pickle.dump(design_dict, fp)   
with open("Desktop/design_dict_NMMSO_MI_P.txt", "rb") as fp:   
    design_dict_NMMSO_MI_P = pickle.load(fp)  

with open("Desktop/fit_dict_NMMSO_MI_P.txt", "wb") as fp:   
    pickle.dump(fit_dict, fp)   
with open("Desktop/fit_dict_NMMSO_MI_P.txt", "rb") as fp:   
    fit_dict_NMMSO_MI_P = pickle.load(fp)     
    

with open("Desktop/x_NMMSO_MI_P.txt", "wb") as fp:   #Pickling
    pickle.dump(x_NMMSO_MI, fp)   
with open("Desktop/x_NMMSO_MI_P.txt", "rb") as fp:   # Unpickling
    x_NMMSO_MI_P = pickle.load(fp)


with open("Desktop/f_NMMSO_MI_P.txt", "wb") as fp:   #Pickling
    pickle.dump(f_NMMSO_MI, fp)   
with open("Desktop/f_NMMSO_MI_P.txt", "rb") as fp:   # Unpickling
    f_NMMSO_MI_P = pickle.load(fp)