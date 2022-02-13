#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 18:08:04 2022

@author: melikedila
"""

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import itertools



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



#%%    Convert data type from double to np.array

dataLD = np.asarray(dataLD)
dataDD = np.asarray(dataDD)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_dataLD = []
norm_dataDD = []


for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD[i]))
    norm_dataDD.append(NormalizeData(dataDD[i]))

norm_dataLD.append(dataLD[-1])
norm_dataDD.append(dataDD[-1])

norm_dataLD_ = [sublist[49:] for sublist in norm_dataLD]
norm_dataDD_ = [sublist[49:] for sublist in norm_dataDD]

    
lst = []
lst.append(norm_dataLD_[0])
lst.append(norm_dataLD_[1])
lst = np.array(lst)
lst= lst.T
df = pd.DataFrame(lst, index =[norm_dataLD_[2][i] for i in range(len(norm_dataLD_[2]))],
                                              columns =['mRNA','P'])   
sns.set(rc={'figure.figsize':(18,8.27)},font_scale = 3,style="whitegrid")   
p = sns.lineplot(data=df)   
sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)#plt.legend(loc='upper left')
p.set(xlabel='time steps', ylabel='normalised expression levels',xlim=(20, 125),ylim=(0,1))
plt.savefig('Desktop/neuro1lp_timeseries_LD.eps', format='eps',bbox_inches='tight')
    
    
#  same for DD

lst = []
lst.append(norm_dataDD_[0])
lst.append(norm_dataDD_[1])
lst = np.array(lst)
lst= lst.T
df = pd.DataFrame(lst, index =[norm_dataDD_[2][i] for i in range(len(norm_dataDD_[2]))],
                                              columns =['mRNA','P'])   
sns.set(rc={'figure.figsize':(18,8.27)},font_scale = 3,style="whitegrid")   
p = sns.lineplot(data=df)   
sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)#plt.legend(loc='upper left')
p.set(xlabel='time steps', ylabel='normalised expression levels',xlim=(20, 125),ylim=(0,1))
plt.savefig('Desktop/neuro1lp_timeseries_normalised_DD.eps', format='eps',bbox_inches='tight')
    
    
    
##################################  Obtaining dat and sol when inputparams given to the function
#####   !) Read all data
read_root = "Desktop/Llyonesse/continuous_fcn_results/"
model_list = {"neuro1lp":2,"neuro2lp": 5, "arabid2lp":8}
optimisers = ["CMAES","NMMSO"]


for model, n_gates in model_list.items():
    for opt in optimisers:
        with open(read_root + model + "/x_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['x_%s_MI_%s' % (opt,model)] = pickle.load(fp)   
        with open(read_root + model + "/f_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['f_%s_MI_%s' % (opt,model)] = pickle.load(fp) 
        if opt =="NMMSO":
            with open(read_root + model + "/fit_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
                globals()['fit_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)   
            with open(read_root + model + "/design_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
                globals()['design_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)           
        gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))      
        for gate in gatesm:
            with open(read_root + model + f"/x_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'x_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp) 
            with open(read_root + model + f"/f_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'f_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp)
            if opt == "NMMSO":
                with open(read_root + model + "/fit_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['fit_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp)   
                with open(read_root + model + "/design_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['design_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp) 
                    
                    
                    

for sol in 

sol_dict = getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5)    


     
    
#%%         ##################3   This is just to fix the file names
gatesm = list(map(list, itertools.product([0,1], repeat=5)))      
for idx,gate in enumerate(gatesm):    
    with open(read_root +f"neuro2lp/x_NMMSO_%s_NMMSO_cts.txt"%gate, "rb") as fp:    #nerede nasil oldugu
        a = pickle.load(fp) 
    with open(read_root +f"neuro2lp/new/x_NMMSO_%s_neuro2lp_cts.txt"%gate, "wb") as fp:   ###  nereye nasil kaydeddecegin
        pickle.dump(a, fp)   
#%%      ###################    bunu unut bile fazlasi vardi onu cikardimm
model= "neuro2lp"
gatesm = list(map(list, itertools.product([0,1], repeat=5)))      
for idx,gate in enumerate(gatesm):  
    with open(read_root + model + f"/f_CMAES_%s_neuro2lp_cts.txt"%gate , "rb") as fp:   
        globals()[f'f_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp) 
        globals()[f'f_%s_%s_%s' % (opt,model,gate)] = globals()[f'f_%s_%s_%s' % (opt,model,gate)] [0:30]
    with open(read_root + model + f"/f_CMAES_%s_neuro2lp_cts.txt"%gate , "wb") as fp:  ###  nereye nasil kaydeddecegin
        pickle.dump(globals()[f'f_%s_%s_%s' % (opt,model,gate)], fp)     
    
#%%  

#################################################################################################################
n_gates = 2 
gatesm = list(map(list, itertools.product([0,1], repeat=n_gates)))   


class Mat_Py:
    def __init__(self):
        self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd = ([] for _ in range(5))
    
    # def update(self, cost, sol_ld, sol_dd, dat_ld, dat_dd):
    #     self.cost.append(cost)
    #     self.sol_ld.append({'x':list(sol_ld['x']._data), 'y':np.asarray(sol_ld['y'], dtype=np.longdouble).tolist()})
    #     self.sol_dd.append({'x':list(sol_dd['x']._data), 'y':list(sol_dd['y']._data)})
    #     self.dat_ld.append({'x':list(dat_ld['x']._data), 'y':list(dat_ld['y']._data)})
    #     self.dat_dd.append({'x':list(sol_dd['x']._data), 'y':list(sol_dd['y']._data)})
        
    def update(self, cost, sol_ld, sol_dd, dat_ld, dat_dd):
        self.cost.append(cost)
        self.sol_ld.append({'x':list(sol_ld['x']._data), 'y':np.asarray(sol_ld['y'], dtype=np.longdouble).tolist()})
        self.sol_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(sol_dd['y'], dtype=np.longdouble).tolist()})
        self.dat_ld.append({'x':list(dat_ld['x']._data), 'y':np.asarray(dat_ld['y'], dtype=np.longdouble).tolist()})
        self.dat_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(dat_dd['y'], dtype=np.longdouble).tolist()})
    
    def get_all(self):
        return [self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd]
         

func_output_CMA = Mat_Py()
func_output_NMMSO = Mat_Py()

gates = gatesm[1]
for idx,sol in enumerate(globals()[f'x_CMAES_neuro1lp_%s' % gates]):   
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_CMA.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))

[cost_CMA, sol_LD_CMA, sol_DD_CMA, datLD_CMA, datDD_CMA] = func_output_CMA.get_all()
# print(cost_values.sol_dd[0])
# list(cost_values.sol_dd[4]['x']._data)
# cost_values.sol_dd[4]['x'][0]._data
# list(cost_values.sol_dd[4]['x'][0]._data)
# print(cost_values.get_all())
####  NMMSO
gates = gatesm[1]
for idx,sol in enumerate(globals()[f'x_NMMSO_neuro1lp_%s' % gates]):   
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_NMMSO.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_NMMSO, sol_LD_NMMSO, sol_DD_NMMSO, datLD_NMMSO, datDD_NMMSO] = func_output_CMA.get_all()    
###############################################################################################################  Plot all time series with 
#%%  


dataLD = np.asarray(dataLD)
dataDD = np.asarray(dataDD)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_dataLD = []
norm_dataDD = []


for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD[i]))
    norm_dataDD.append(NormalizeData(dataDD[i]))

norm_dataLD.append(dataLD[-1])
norm_dataDD.append(dataDD[-1])

norm_dataLD_ = [sublist[49:] for sublist in norm_dataLD]
norm_dataDD_ = [sublist[49:] for sublist in norm_dataDD]

    
lst = []
lst.append(norm_dataLD_[0])
lst.append(norm_dataLD_[1])
lst = np.array(lst)
lst= lst.T
df = pd.DataFrame(lst, index =[norm_dataLD_[2][i] for i in range(len(norm_dataLD_[2]))],
                                              columns =['mRNA','P'])   
sns.set(rc={'figure.figsize':(18,8.27)},font_scale = 3,style="whitegrid")   
p = sns.lineplot(data=df)   
sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)#plt.legend(loc='upper left')
p.set(xlabel='time steps', ylabel='normalised expression levels',xlim=(20, 125),ylim=(0,1))
plt.savefig('Desktop/neuro1lp_timeseries_LD.eps', format='eps',bbox_inches='tight')

print(sol_LD_NMMSO)
print(sol_LD_CMA)




























































