#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:59:54 2022

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

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
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




#%%

for k in range(152,153):
    gate = gatesm[k]
    c = 0
    def arabid2lp_gates(inputparams):
        init = inputparams
        inputparams = inputparams[0:15]
        inputparams = list(inputparams)
        inputparams = matlab.double([inputparams])
        gates = init[15:24]
        gates = list(gates)
        gates = matlab.double([gates])
        cost=eng.getBoolCost_arabid2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        return cost

    def arabid2lp(inputparams):
        dist = []
        if (inputparams[0] + inputparams[1] + inputparams[2] >= 24) :
            dist.append(inputparams[0] + inputparams[1] + inputparams[2] - 23.5)
            
        if inputparams[4] + inputparams[5] >= 24:
            dist.append(inputparams[4] + inputparams[5] - 23.5)
            
        if inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] >= 24:
            dist.append(inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] - 23.5)  
        
        distance = sum(dist)
        for i in range(8):
            inputparams[i] = round(inputparams[i])
        #cost=arabid2lp_gates(inputparams)
        cost = arabid2lp_gates(inputparams) + distance
            
        return cost     #


#####################################################################################   NMMSO-MI + Penalty
#%%  

    f_NMMSO_MI = []
    x_NMMSO_MI = []
    
    
        
    class MyProblem:
      @staticmethod
      def fitness(inputparams):
        for i in inputparams:
           cost= arabid2lp(inputparams)     
           return - cost
        
      @staticmethod
      def get_bounds():
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [24,24,24,24,24,24,12,12,12,1,1,1,1,4,4,1,1,1,1,1,1,1,1]
        
    
    design_dict = {}
    fit_dict = {}
    for i in range(30):
        start_time = time.time()
        local_modes =[]
        local_fit = []
        def main():
            number_of_fitness_evaluations = 5*10**3
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
    for key,value in fit_dict.items():
        f = max(value)
        f_NMMSO_MI.append(f)
        id = value.index(f)
        l =  list(design_dict[key][id])
        x_NMMSO_MI.append(l)


    with open("Desktop/design_dict_NMMSO_MI_arabid2lp.txt", "wb") as fp:   
        pickle.dump(design_dict, fp)   
    with open("Desktop/design_dict_NMMSO_MI_arabid2lp.txt", "rb") as fp:   
        design_dict_NMMSO_MI_P = pickle.load(fp)  

    with open("Desktop/fit_dict_NMMSO_MI_arabid2lp.txt", "wb") as fp:   
        pickle.dump(fit_dict, fp)   
    with open("Desktop/fit_dict_NMMSO_MI_arabid2lp..txt", "rb") as fp:   
        fit_dict_NMMSO_MI_P = pickle.load(fp)     
        

    with open("Desktop/x_NMMSO_MI_arabid2lp.txt", "wb") as fp:   #Pickling
        pickle.dump(x_NMMSO_MI, fp)   
    with open("Desktop/x_NMMSO_MI_arabid2lp.txt", "rb") as fp:   # Unpickling
        x_NMMSO_MI_P = pickle.load(fp)


    with open("Desktop/f_NMMSO_MI_arabid2lp.txt", "wb") as fp:   #Pickling
        pickle.dump(f_NMMSO_MI, fp)   
    with open("Desktop/f_NMMSO_MI_arabid2lp.txt", "rb") as fp:   # Unpickling
        f_NMMSO_MI_P = pickle.load(fp)