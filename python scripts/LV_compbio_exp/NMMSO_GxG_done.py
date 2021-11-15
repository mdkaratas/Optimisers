
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:19:38 2021

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

design_dict = {}
fit_dict = {}
for i in range(30):
    start_time = time.time()
    local_modes =[]
    local_fit = []
    def main():
        number_of_fitness_evaluations  = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            local_modes.append(mode_result.location)
            local_fit.append(mode_result.value)
        design_dict[i] = local_modes
        fit_dict[i] = local_fit
        print("--- %s seconds ---" % (time.time() - start_time))   
    if __name__ == "__main__":
        main()


x_NMMSO_1 = []
f_NMMSO_1 = []
k = 0
for key,value in fit_dict.items():
    f = max(value)
    f_NMMSO_1.append(f)
    id = value.index(f)
    l =  list(design_dict[key][id])
    x_NMMSO_1.append(l)
    
    


 
with open("design_dict_NMMSO1.txt", "wb") as fp:   
    pickle.dump(design_dict, fp)   
with open("Desktop/design_dict_NMMSO1.txt", "rb") as fp:   
    design_dict_NMMSO1 = pickle.load(fp)  

with open("Desktop/fit_dict_NMMSO1.txt", "wb") as fp:   
    pickle.dump(fit_dict, fp)   
with open("Desktop/fit_dict_NMMSO1.txt", "rb") as fp:   
    fit_dict_NMMSO1 = pickle.load(fp)      
        
with open("Desktop/x_NMMSO_1.txt", "wb") as fp:   
    pickle.dump(x_NMMSO_1, fp)   
with open("Desktop/x_NMMSO_1.txt", "rb") as fp:   
    x_NMMSO_1 = pickle.load(fp)  


with open("Desktop/f_NMMSO_1.txt", "wb") as fp:  
    pickle.dump(f_NMMSO_1, fp)   
with open("Desktop/f_NMMSO_1.txt", "rb") as fp:   
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

design_dict = {}
fit_dict = {}
for i in range(30):
    start_time = time.time()
    local_modes =[]
    local_fit = []
    def main():
        number_of_fitness_evaluations  = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            local_modes.append(mode_result.location)
            local_fit.append(mode_result.value)
        design_dict[i] = local_modes
        fit_dict[i] = local_fit
        print("--- %s seconds ---" % (time.time() - start_time))   
    if __name__ == "__main__":
        main()


x_NMMSO_2 = []
f_NMMSO_2 = []
k = 0
for key,value in fit_dict.items():
    f = max(value)
    f_NMMSO_2.append(f)
    id = value.index(f)
    l =  list(design_dict[key][id])
    x_NMMSO_2.append(l)
    
    


 
with open("design_dict_NMMSO2.txt", "wb") as fp:   
    pickle.dump(design_dict, fp)   
with open("Desktop/design_dict_NMMSO2.txt", "rb") as fp:   
    design_dict_NMMSO2 = pickle.load(fp)  

with open("Desktop/fit_dict_NMMSO2.txt", "wb") as fp:   
    pickle.dump(fit_dict, fp)   
with open("Desktop/fit_dict_NMMSO2.txt", "rb") as fp:   
    fit_dict_NMMSO2 = pickle.load(fp)      
        
with open("Desktop/x_NMMSO_2.txt", "wb") as fp:   
    pickle.dump(x_NMMSO_2, fp)   
with open("Desktop/x_NMMSO_2.txt", "rb") as fp:   
    x_NMMSO_2 = pickle.load(fp)  


with open("Desktop/f_NMMSO_2.txt", "wb") as fp:  
    pickle.dump(f_NMMSO_2, fp)   
with open("Desktop/f_NMMSO_2.txt", "rb") as fp:   
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

design_dict = {}
fit_dict = {}
for i in range(30):
    start_time = time.time()
    local_modes =[]
    local_fit = []
    def main():
        number_of_fitness_evaluations  = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            local_modes.append(mode_result.location)
            local_fit.append(mode_result.value)
        design_dict[i] = local_modes
        fit_dict[i] = local_fit
        print("--- %s seconds ---" % (time.time() - start_time))   
    if __name__ == "__main__":
        main()


x_NMMSO_3 = []
f_NMMSO_3 = []
k = 0
for key,value in fit_dict.items():
    f = max(value)
    f_NMMSO_3.append(f)
    id = value.index(f)
    l =  list(design_dict[key][id])
    x_NMMSO_3.append(l)
    
    


 
with open("design_dict_NMMSO3.txt", "wb") as fp:   
    pickle.dump(design_dict, fp)   
with open("Desktop/design_dict_NMMSO3.txt", "rb") as fp:   
    design_dict_NMMSO3 = pickle.load(fp)  

with open("Desktop/fit_dict_NMMSO3.txt", "wb") as fp:   
    pickle.dump(fit_dict, fp)   
with open("Desktop/fit_dict_NMMSO3.txt", "rb") as fp:   
    fit_dict_NMMSO3 = pickle.load(fp)      
        
with open("Desktop/x_NMMSO_3.txt", "wb") as fp:   
    pickle.dump(x_NMMSO_3, fp)   
with open("Desktop/x_NMMSO_3.txt", "rb") as fp:   
    x_NMMSO_3 = pickle.load(fp)  


with open("Desktop/f_NMMSO_3.txt", "wb") as fp:  
    pickle.dump(f_NMMSO_3, fp)   
with open("Desktop/f_NMMSO_3.txt", "rb") as fp:   
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

design_dict = {}
fit_dict = {}
for i in range(30):
    start_time = time.time()
    local_modes =[]
    local_fit = []
    def main():
        number_of_fitness_evaluations  = 5*10**3
        problem = UniformRangeProblem(MyProblem())    
        nmmso = Nmmso(problem)
        my_result = nmmso.run(number_of_fitness_evaluations)
        for mode_result in my_result:
            local_modes.append(mode_result.location)
            local_fit.append(mode_result.value)
        design_dict[i] = local_modes
        fit_dict[i] = local_fit
        print("--- %s seconds ---" % (time.time() - start_time))   
    if __name__ == "__main__":
        main()


x_NMMSO_0 = []
f_NMMSO_0 = []
k = 0
for key,value in fit_dict.items():
    f = max(value)
    f_NMMSO_0.append(f)
    id = value.index(f)
    l =  list(design_dict[key][id])
    x_NMMSO_0.append(l)
    
    


 
with open("design_dict_NMMSO0.txt", "wb") as fp:   
    pickle.dump(design_dict, fp)   
with open("Desktop/design_dict_NMMSO0.txt", "rb") as fp:   
    design_dict_NMMSO0 = pickle.load(fp)  

with open("Desktop/fit_dict_NMMSO0.txt", "wb") as fp:   
    pickle.dump(fit_dict, fp)   
with open("Desktop/fit_dict_NMMSO0.txt", "rb") as fp:   
    fit_dict_NMMSO0 = pickle.load(fp)      
        
with open("Desktop/x_NMMSO_0.txt", "wb") as fp:   
    pickle.dump(x_NMMSO_0, fp)   
with open("Desktop/x_NMMSO_0.txt", "rb") as fp:   
    x_NMMSO_0 = pickle.load(fp)  


with open("Desktop/f_NMMSO_0.txt", "wb") as fp:  
    pickle.dump(f_NMMSO_0, fp)   
with open("Desktop/f_NMMSO_0.txt", "rb") as fp:   
    f_NMMSO_0 = pickle.load(fp)  


##########
with open("Desktop/f_NMMSO_0.txt", "rb") as fp:   
    f_NMMSO_0 = pickle.load(fp)  

with open("Desktop/f_NMMSO_2.txt", "rb") as fp:   
    f_NMMSO_2 = pickle.load(fp)  
    

with open("Desktop/f_NMMSO_3.txt", "rb") as fp:   
    f_NMMSO_3 = pickle.load(fp)      


#######################################  ####################



gate_0 = [-1*i for i in f_NMMSO_0]
gate_1 = [-1*i for i in f_NMMSO_1]
gate_2 = [-1*i for i in f_NMMSO_2]
gate_3 = [-1*i for i in f_NMMSO_3]



###  boxplot

data = [gate_1, gate_2,gate_0,gate_3]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '2', '0',
                    '3'])
plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("NMMSO: G x G")
plt.show()
