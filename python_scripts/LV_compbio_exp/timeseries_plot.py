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
import matplotlib


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

#%%   This will be brought to on function format --- check it

def LD_routine(tot_hours):

    time_swap  = []      # this is to find all 6-18 light switches
    tot_hours = 120
    for i in range(0,tot_hours+2):
        if i % 2 == 1:
            time_swap.append(6*i)
    
    def partition(values, indices):    # this func is for dividing a list based on another list
        idx = 0
        for index in indices:
            sublist = []
            while idx < len(values) and values[idx] < index:
                sublist.append(values[idx])
                idx += 1
            if sublist:
                yield sublist
    
    df = {'time': norm_dataLD[2],
      'A':norm_dataLD[0],
      'B':norm_dataLD[1]}    
    dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list
    
    
    a = []
    b = []
    for i in range(len(dl_sep)):
        if i % 2 == 0:
            a.append(dl_sep[i])
        elif i % 2 == 1:
            b.append(dl_sep[i])
    A =[]
    B=[]
    for i in range(len(a)):
        A.append([a[i][0],a[i][-1]])
    
    for i in range(len(b)):
        B.append([b[i][0],b[i][-1]])
        
        return A,B

#%%    Normalising the data func
  
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


#%%    Convert data type from double to np.array

dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)


######################################################################################################  su bi dursun calistirma
norm_dataLD = []
norm_dataDD = []


for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD_arr[i]))
    norm_dataDD.append(NormalizeData(dataDD_arr[i]))

norm_dataLD.append(dataLD_arr[-1])
norm_dataDD.append(dataDD_arr[-1])

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
    
####################################################################################################################    
    
##################################  Obtaining dat and sol when inputparams given to the function
#####   !) Read all data
read_root = "Desktop/Llyonesse/continuous_fcn_results/"
model_list = {"neuro1lp":2,"neuro2lp": 5, "arabid2lp":8}
optimisers = ["CMAES","NMMSO"]
#optimisers = ["NMMSO"]


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
        for gate in gatesm: #[3:5]
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
gatesm = list(map(list, itertools.product([0,1], repeat=8)))      
for idx,gate in enumerate(gatesm): 
    gate = gatesm[253]
    with open(read_root +f"arabid2lp/f_CMAES_%s_cts.txt"%gate, "rb") as fp:    #nerede nasil oldugu
        a = pickle.load(fp) 
    with open(read_root +f"arabid2lp/new/f_CMAES_%s_arabid2lp_cts.txt"%gate, "wb") as fp:   ###  nereye nasil kaydeddecegin
        pickle.dump(a, fp)
        
gatesm = list(map(list, itertools.product([0,1], repeat=8)))      
for idx,gate in enumerate(gatesm):    
    with open(read_root +f"arabid2lp/x_NMMSO_%s_NMMSO_cts.txt"%gate, "rb") as fp:    #nerede nasil oldugu
        a = pickle.load(fp) 
    with open(read_root +f"arabid2lp/new/x_NMMSO_%s_arabid2lp_cts.txt"%gate, "wb") as fp:   ###  nereye nasil kaydeddecegin
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

gate = gatesm[1]
for idx,sol in enumerate(globals()[f'x_CMAES_neuro1lp_%s' % gate]):  
    gates = list(gate)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    print(inputparams)
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
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_NMMSO.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_NMMSO, sol_LD_NMMSO, sol_DD_NMMSO, datLD_NMMSO, datDD_NMMSO] = func_output_CMA.get_all()    
###############################################################################################################  Plot all time series with 
#%%  
########### normalise data

dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_dataLD = []
norm_dataDD = []

for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD_arr[i]))
    norm_dataDD.append(NormalizeData(dataDD_arr[i]))

norm_dataLD.append(dataLD_arr[-1])
norm_dataDD.append(dataDD_arr[-1])

norm_dataLD= [sublist[48:] for sublist in norm_dataLD]    ##### simply add or remove this based on where to start/end the graphs 24 or 0 ?
norm_dataDD= [sublist[48:] for sublist in norm_dataDD]
#%%  
########### the same as above but total normalisation of states across variables
dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)
norm_dataLD = []
norm_dataDD = []
for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD_arr[i]))
    norm_dataDD.append(NormalizeData(dataDD_arr[i]))
norm_dataLD= [sublist[48:] for sublist in norm_dataLD]    ##### simply add or remove this based on where to start/end the graphs 24 or 0 ?
norm_dataDD= [sublist[48:] for sublist in norm_dataDD]
n_total_norm = [norm_dataLD[0] ,norm_dataLD[1] ,norm_dataDD[0] ,norm_dataDD[1]]
result = []     # lidt to accumulate all 2 vars for LD and 2 for DD so that
[ result.extend(el) for el in n_total_norm] 
result = np.array([result]).transpose()

norm_data = NormalizeData(result)
    
    
norm_dataLD[0] = norm_data[0:193]
norm_dataLD[1] = norm_data[193:386]
norm_dataDD[0] = norm_data[386:559]
norm_dataDD[1] = norm_data[559:732]

norm_dataLD.append(dataLD_arr[-1][48:])
norm_dataDD.append(dataDD_arr[-1][48:])

################################### new noramlisation--check
normalize= []  #336

for i in dataLD_arr[0]:
    normalize.append(i)
for i in dataLD_arr[1]:
    normalize.append(i)   
for i in dataDD_arr[0]:
    normalize.append(i)
for i in dataDD_arr[1]:
    normalize.append(i) 
N = NormalizeData(normalize)

no_dataLD_arr = []
no_dataDD_arr = []
no_dataLD_arr.append(N[0:241])
no_dataLD_arr.append(N[241:482])
no_dataDD_arr.append(N[482:703])
no_dataDD_arr.append(N[703:924])
no_dataLD_arr[0] = no_dataLD_arr[0][48:]
no_dataLD_arr[1] = no_dataLD_arr[1][48:]
no_dataDD_arr[0] = no_dataDD_arr[0][48:]
no_dataDD_arr[1] = no_dataDD_arr[1][48:]
no_dataLD_arr.append(dataLD_arr[-1][48:])
no_dataDD_arr.append(dataDD_arr[-1][48:])
############  burada bastan--karsilastir

###########################################################   light switches-- timeseries plot framework

time_swap  = []      # this is to find all 6-18 light switches
tot_hours = 120
for i in range(0,tot_hours+2):
    if i % 2 == 1:
        time_swap.append(6*i)

def partition(values, indices):    # this func is for dividing a list based on another list
    idx = 0
    for index in indices:
        sublist = []
        while idx < len(values) and values[idx] < index:
            sublist.append(values[idx])
            idx += 1
        if sublist:
            yield sublist

df = {'time': no_dataLD_arr[2],
  'A':no_dataLD_arr[0],
  'B':no_dataLD_arr[1]}    
dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list



a = []
b = []
for i in range(len(dl_sep)):
    if i % 2 == 0:
        a.append(dl_sep[i])
    elif i % 2 == 1:
        b.append(dl_sep[i])
A =[]
B=[]
for i in range(len(a)):
    #print([a[i][0],a[i][-1]])
    A.append([a[i][0],a[i][-1]])

for i in range(len(b)):

    B.append([b[i][0],b[i][-1]])
    

#import matplotlib.ticker as ticker
# plt.plot(norm_dataLD[2], norm_dataLD[0])
# plt.plot(norm_dataLD[2], norm_dataLD[1])
matplotlib.rc('figure', figsize=(20, 7))
l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(np.arange(0, 1.1, 0.2),fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),
          fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='gray')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/LD_ts_data_norm_neuro1lp.eps', format='eps')
plt.show()



#####################################################################################  now plot CMAES timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/LD_ts_CMA_neuro1lp.eps', format='eps')    
plt.show()
#plt.savefig('x.png')

#####################################################################################  now plot NMMSO timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0] #this indexfr the state of the first variable
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD

for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')   
plt.savefig('Desktop/LD_ts_NMMSO_neuro1lp.eps', format='eps')        
plt.show()
#plt.savefig('x.png')

######################################################################################################################### plot DD- data

df = {'time': no_dataDD_arr[2],
  'A':no_dataDD_arr[0],
  'B':no_dataDD_arr[1]}   

matplotlib.rc('figure', figsize=(20, 7))
l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(np.arange(0, 1.1, 0.2),fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),
          fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

plt.axvspan(24,120, facecolor='gray')

plt.savefig('Desktop/DD_ts_data_norm_neuro1lp.eps', format='eps')
plt.show()

#####################################################################################  now plot CMAES timeseries DD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

plt.axvspan(24,120, facecolor='gray')
plt.savefig('Desktop/DD_ts_CMA_neuro1lp.eps', format='eps')    
plt.show()
#plt.savefig('x.png')

#####################################################################################  now plot NMMSO timeseries DD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0] #this indexfr the state of the first variable
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD

for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

plt.axvspan(24,120, facecolor='gray')
plt.savefig('Desktop/DD_ts_NMMSO_neuro1lp.eps', format='eps')        
plt.show()
#plt.savefig('x.png')
############################################################################################################  2nd model




###########################  burasi duzenle   
### HEATMAP
# t = [7.06376295173869,3.7889859616007584,8.024697492449109,0.45292277509422907,0.569881816090693]

# c = [3.7795606498167045,6.995704083896072,10.884323404862542,0.3457868627036032,0.61080069736405]

#########################################  HEATMAP bu 1. model icin :)---   1298 git

gate = gatesm[1]



t = globals()[f'x_NMMSO_neuro1lp_%s' % gate][4]
c = globals()[f'x_NMMSO_neuro1lp_%s' % gate][7]

t = [[]]
t[0] = np.array(globals()[f'x_NMMSO_neuro1lp_%s' % gate][4])
t.append(np.array(globals()[f'x_NMMSO_neuro1lp_%s' % gate][7]))

func_output_t = Mat_Py()
for idx,sol in enumerate(t):   
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_t.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_t, sol_LD_t, sol_DD_t, datLD_t, datDD_t] = func_output_t.get_all()    



############   24den sonraki hal--- burayi duzenle-- 24 yoksa bir oncekini alsin ama 24 varsa oradan itibaren alsin, cunku data 24den baslayacak

datLD_t[0]['x'] = datLD_t[0]['x'][4:23]
datLD_t[0]['y'][0] = datLD_t[0]['y'][0][4:23]
datLD_t[0]['y'][1] = datLD_t[0]['y'][1][4:23]

datLD_t[1]['x'] = datLD_t[1]['x'][4:23]
datLD_t[1]['y'][0] = datLD_t[1]['y'][0][4:23]
datLD_t[1]['y'][1] = datLD_t[1]['y'][1][4:23]

###################################################  altta fnc yazilmisi var

### alttakini fonk olarak yaz    ------ data 1-data 2 birlestirici fonk yaz !  hem LD icin hem DD icin !
solution = sol_LD_t
data_threshold = datLD_t
data1={}
#data1['x'] = data1['x'] + 3
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[0]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[0]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[0]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[0]['y'][1][j-1])
            y = [y0,y1]               
            break
data1['x'] = time
data1['y']= y

### alttakini fonk olarak yaz
data2={}
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[1]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[1]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[1]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[1]['y'][1][j-1])
            y = [y0,y1]               
            break
data2['x'] = time
data2['y']= y

light_cbar = [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
#############################################  HEATMAP    --- time equalise

#############################################
###########  yine genel fonk yapilmayi bekleyen asagisi: verilen data araliginda stateleri duzwnliyor--24den 180e adar 8 aralikli yapmak istiyorsam:

### zamanlarina bakip value'lerini olusturuyoruz ---ikinci variable aggregate edilecek olan list time list tekrar


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      

####### comne tout asagi        
#time_list = [6*i for i in range(4,21)]    # this will be used for labelling the heatmap
# time_list = []
# for i in range(4,21) :
#     if i==4 or i == 20:
#         time_list.append(6*i)
#     if i % 2 == 1:
#         time_list.append(6*i)
# aggregated_data = datLD_t
# first_data = time_equalise(time_list,aggregated_data,0)   
# second_data = time_equalise(time_list,aggregated_data,1)   

# solution = sol_LD_t
# first_solution = time_equalise(time_list,solution,0)   
# second_solution = time_equalise(time_list,solution,1)   
  
# harvest = np.empty([9,len(time_list)])
# harvest[0] = light_cbar
# harvest[1] = first_data[0]
# harvest[2] = first_solution[0]
# harvest[3] = second_data[0]
# harvest[4] = second_solution[0]
# harvest[5] = first_data[1]
# harvest[6] = first_solution[1]
# harvest[7] = second_data[1]
# harvest[8] = second_solution[1]

# y_axis_labels = [[],'mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
# cols = time_list

# df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
#                    index = y_axis_labels)

##############  this is for LD colormap

light_cbar = [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
##################


matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
sns.heatmap(df,xticklabels=True, yticklabels=True,cmap="Reds",cbar = False)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
#s.set_xticklabels(labels = x_axis_labels,rotation=90)


########################################################   0.5 fark yapiyor o arada olani gostermek zorundasin... here we go-last



##############  this is for LD colormap
time_list = np.linspace(24,120,193)
def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0 = []
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data['x']):         
            if val<=value:
                y0.append(aggregated_data['y'][j-1])
                y = y0             
                break                 
    while len(time_list)!= len(y0):
        y.append(aggregated_data['y'][-1])
    return y  

light_switch= []
for i in range(4,21) :
    if i == 4:
        light_switch.append(6*i)    
    if i % 2 == 1:
        light_switch.append(6*i)    
light_cbar = {}
light_cbar['x'] = light_switch
light_cbar['y'] = [1,0,1,0,1,0,1,0,1]  
T = time_equalise(time_list,light_cbar,m=0)
##################


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      
cb = []
for i in range(len(T)):
    cb.append(float(T[i]))
    



aggregated_data = datLD_t
first_data = time_equalise(time_list,aggregated_data,0)   
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_LD_t
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)],dtype=object)
harvest[0] = cb
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]

y_axis_labels = ['Light regime(LD/DD)','mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
                   index = y_axis_labels,dtype="float")


plt.figure()
matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
s = sns.heatmap(df,xticklabels=True, yticklabels=True,cmap='PuRd',cbar = False)  ### Reds, Greys,Blues,

plt.xticks(rotation=0)
for ind, label in enumerate(s.get_xticklabels()):
    if ind % 16 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)    
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=False,left=True) 
plt.locator_params(axis='x', nbins=200)  

#plt.tick_params(axis=u'both',length=15,color='black',bottom=True,left=True)                                               
plt.tight_layout() 



############################################################################################################## HERE PLOT
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

gate = gatesm[1]
for idx,sol in enumerate(globals()[f'x_CMAES_neuro1lp_%s' % gate]):  
    gates = list(gate)
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
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_NMMSO.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_NMMSO, sol_LD_NMMSO, sol_DD_NMMSO, datLD_NMMSO, datDD_NMMSO] = func_output_CMA.get_all()    


################################### new noramlisation--check

dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)



normalize= []  #336

for i in dataLD_arr[0]:
    normalize.append(i)
for i in dataLD_arr[1]:
    normalize.append(i)   
for i in dataDD_arr[0]:
    normalize.append(i)
for i in dataDD_arr[1]:
    normalize.append(i) 
N = NormalizeData(normalize)

no_dataLD_arr = []
no_dataDD_arr = []
no_dataLD_arr.append(N[0:241])
no_dataLD_arr.append(N[241:482])
no_dataDD_arr.append(N[482:703])
no_dataDD_arr.append(N[703:924])
no_dataLD_arr[0] = no_dataLD_arr[0][48:]
no_dataLD_arr[1] = no_dataLD_arr[1][48:]
no_dataDD_arr[0] = no_dataDD_arr[0][48:]
no_dataDD_arr[1] = no_dataDD_arr[1][48:]
no_dataLD_arr.append(dataLD_arr[-1][48:])
no_dataDD_arr.append(dataDD_arr[-1][48:])

time_swap  = []      # this is to find all 6-18 light switches
tot_hours = 120
for i in range(0,tot_hours+2):
    if i % 2 == 1:
        time_swap.append(6*i)

def partition(values, indices):    # this func is for dividing a list based on another list
    idx = 0
    for index in indices:
        sublist = []
        while idx < len(values) and values[idx] < index:
            sublist.append(values[idx])
            idx += 1
        if sublist:
            yield sublist

df = {'time': no_dataLD_arr[2],
  'A':no_dataLD_arr[0],
  'B':no_dataLD_arr[1]}    
dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list



a = []
b = []
for i in range(len(dl_sep)):
    if i % 2 == 0:
        a.append(dl_sep[i])
    elif i % 2 == 1:
        b.append(dl_sep[i])
A =[]
B=[]
for i in range(len(a)):
    #print([a[i][0],a[i][-1]])
    A.append([a[i][0],a[i][-1]])

for i in range(len(b)):

    B.append([b[i][0],b[i][-1]])
    


#####################################################################################  now plot CMAES timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
#plt.savefig('Desktop/LD_ts_CMA_neuro1lp.eps', format='eps')    
plt.show()
#plt.savefig('x.png')

#####################################################################################  now plot NMMSO timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0] #this indexfr the state of the first variable
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD

for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')   
plt.savefig('Desktop/LD_ts_NMMSO_neuro1lp.eps', format='eps')        
plt.show()
#plt.savefig('x.png')

##############################################################################################


########################################################   0.5 fark yapiyor o arada olani gostermek zorundasin... here we go-last



##############  this is for LD colormap
time_list = np.linspace(24,120,193)
def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0 = []
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data['x']):         
            if val<=value:
                y0.append(aggregated_data['y'][j-1])
                y = y0             
                break                 
    while len(time_list)!= len(y0):
        y.append(aggregated_data['y'][-1])
    return y  

light_switch= []
for i in range(4,21) :
    if i == 4:
        light_switch.append(6*i)    
    if i % 2 == 1:
        light_switch.append(6*i)    
light_cbar = {}
light_cbar['x'] = light_switch
light_cbar['y'] = [1,0,1,0,1,0,1,0,1]  
T = time_equalise(time_list,light_cbar,m=0)
##################


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      
cb = []
for i in range(len(T)):
    cb.append(float(T[i]))
    



aggregated_data = datLD_t
first_data = time_equalise(time_list,aggregated_data,0)   
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_LD_t
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)],dtype=object)
harvest[0] = cb
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]

y_axis_labels = ['Light regime(LD/DD)','mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
                   index = y_axis_labels,dtype="float")


plt.figure()
matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
s = sns.heatmap(df,xticklabels=True, yticklabels=True,cmap='PuRd',cbar = False)  ### Reds, Greys,Blues,

plt.xticks(rotation=0)
for ind, label in enumerate(s.get_xticklabels()):
    if ind % 16 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)    
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=False,left=True) 
plt.locator_params(axis='x', nbins=200)  

#plt.tick_params(axis=u'both',length=15,color='black',bottom=True,left=True)                                               
plt.tight_layout() 



#########################################
n_gates = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
opt =  "CMAES" #"NMMSO" # 
model = "neuro2lp" #,"neuro2lp": 5}
gate = gatesm[6]


a = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
a[6]
p = [4.631808830916931, 1.8612076130496307, 3.0850310072931886, 11.931720288151764,11.999998198637584,0.21995975162031622,0.3664454122358184,0.14753899751818347]
c = [4.632088076074845,1.8614272416316369,9.160975085260269,5.8550589821900445,11.99998867231914,0.21996291321663813,0.366429814651292,0.1475669550730162]
###########################  burasi duzenle   
### HEATMAP
t = [7.06376295173869,3.7889859616007584,8.024697492449109,0.45292277509422907,0.569881816090693]

c = [3.7795606498167045,6.995704083896072,10.884323404862542,0.3457868627036032,0.61080069736405]


#ax = sns.heatmap(df)
  for idx,sol in enumerate(globals()[f'x_NMMSO_%s_%s' % gates]): 
      #print(sol)
      gates = list(gates)

t = [[]]
t[0] = np.array(p)
t.append(np.array(c))

func_output_t = Mat_Py()
for idx,sol in enumerate(t):   
    #print(sol)
    gates = list(gate)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    print(inputparams)
    func_output_t.update(*eng.getBoolCost_cts_neuro2lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_t, sol_LD_t, sol_DD_t, datLD_t, datDD_t] = func_output_t.get_all()    



############   24den sonraki hal--- burayi duzenle-- 24 yoksa bir oncekini alsin ama 24 varsa oradan itibaren alsin, cunku data 24den baslayacak

datLD_t[0]['x'] = datLD_t[0]['x'][4:23]
datLD_t[0]['y'][0] = datLD_t[0]['y'][0][4:23]
datLD_t[0]['y'][1] = datLD_t[0]['y'][1][4:23]

datLD_t[1]['x'] = datLD_t[1]['x'][4:23]
datLD_t[1]['y'][0] = datLD_t[1]['y'][0][4:23]
datLD_t[1]['y'][1] = datLD_t[1]['y'][1][4:23]

###################################################  altta fnc yazilmisi var

### alttakini fonk olarak yaz    ------ data 1-data 2 birlestirici fonk yaz !  hem LD icin hem DD icin !
solution = sol_LD_t
data_threshold = datLD_t
data1={}
#data1['x'] = data1['x'] + 3
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[0]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[0]['x']):         
        if val<=value:
            print(val,value)
            y0.append(data_threshold[0]['y'][0][j-1])
            print(y0)
            y1.append(data_threshold[0]['y'][1][j-1])
            y = [y0,y1]               
            break
data1['x'] = time
data1['y']= y

### alttakini fonk olarak yaz
data2={}
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[1]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[1]['x']):         
        if val<=value:
            print(val,value)
            y0.append(data_threshold[1]['y'][0][j-1])
            print(y0)
            y1.append(data_threshold1][1]['y'][1][j-1])
            y = [y0,y1]               
            break
data2['x'] = time
data2['y']= y


#############################################  HEATMAP    --- time equalise

#############################################
###########  yine genel fonk yapilmayi bekleyen asagisi: verilen data araliginda stateleri duzwnliyor--24den 180e adar 8 aralikli yapmak istiyorsam:

### zamanlarina bakip value'lerini olusturuyoruz ---ikinci variable aggregate edilecek olan list time list tekrar


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      

        
#time_list = [6*i for i in range(4,21)]    # this will be used for labelling the heatmap
time_list = []
for i in range(4,21) :
    if i==4 or i == 20:
        time_list.append(6*i)
    if i % 2 == 1:
        time_list.append(6*i)
aggregated_data = datLD_t
first_data = time_equalise(time_list,aggregated_data,0)   
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_LD_t
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)])
harvest[0] = light_cbar
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]

y_axis_labels = [[],'mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
                   index = y_axis_labels)

##############  this is for LD colormap

light_cbar = [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
##################


matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
sns.heatmap(df,xticklabels=True, yticklabels=True,cmap="Reds",cbar = False)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
#s.set_xticklabels(labels = x_axis_labels,rotation=90)


########################################################   0.5 fark yapiyor o arada olani gostermek zorundasin... here we go-last


######################################################    LASTTT 

gate = gatesm[1]



t = globals()[f'x_NMMSO_neuro1lp_%s' % gate][4]
c = globals()[f'x_NMMSO_neuro1lp_%s' % gate][7]

t = [[]]
t[0] = np.array(globals()[f'x_NMMSO_neuro1lp_%s' % gate][4])
t.append(np.array(globals()[f'x_NMMSO_neuro1lp_%s' % gate][7]))

func_output_t = Mat_Py()
for idx,sol in enumerate(t):   
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_t.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_t, sol_LD_t, sol_DD_t, datLD_t, datDD_t] = func_output_t.get_all()    



############   24den sonraki hal--- burayi duzenle-- 24 yoksa bir oncekini alsin ama 24 varsa oradan itibaren alsin, cunku data 24den baslayacak

datLD_t[0]['x'] = datLD_t[0]['x'][4:23]
datLD_t[0]['y'][0] = datLD_t[0]['y'][0][4:23]
datLD_t[0]['y'][1] = datLD_t[0]['y'][1][4:23]

datLD_t[1]['x'] = datLD_t[1]['x'][4:23]
datLD_t[1]['y'][0] = datLD_t[1]['y'][0][4:23]
datLD_t[1]['y'][1] = datLD_t[1]['y'][1][4:23]

###################################################  altta fnc yazilmisi var

### alttakini fonk olarak yaz    ------ data 1-data 2 birlestirici fonk yaz !  hem LD icin hem DD icin !
solution = sol_LD_t
data_threshold = datLD_t
data1={}
#data1['x'] = data1['x'] + 3
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[0]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[0]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[0]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[0]['y'][1][j-1])
            y = [y0,y1]               
            break
data1['x'] = time
data1['y']= y

### alttakini fonk olarak yaz
data2={}
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[1]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[1]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[1]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[1]['y'][1][j-1])
            y = [y0,y1]               
            break
data2['x'] = time
data2['y']= y

light_cbar = [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
##############  this is for LD colormap
time_list = np.linspace(24,120,193)
# asagisi tekrar comt out
def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0 = []
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data['x']):         
            if val<=value:
                y0.append(aggregated_data['y'][j-1])
                y = y0             
                break                 
    while len(time_list)!= len(y0):
        y.append(aggregated_data['y'][-1])
    return y  

light_switch= []
for i in range(4,21) :
    if i == 4:
        light_switch.append(6*i)    
    if i % 2 == 1:
        light_switch.append(6*i)    
light_cbar = {}
light_cbar['x'] = light_switch
light_cbar['y'] = [1,0,1,0,1,0,1,0,1]  
T = time_equalise(time_list,light_cbar,m=0)
##################


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      
cb = []
for i in range(len(T)):
    cb.append(float(T[i]))
    



aggregated_data = datLD_t
first_data = time_equalise(time_list,aggregated_data,0)   
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_LD_t
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)],dtype=object)
harvest[0] = cb
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]

y_axis_labels = ['Light regime(LD/DD)','mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
                   index = y_axis_labels,dtype="float")


plt.figure()
matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
s = sns.heatmap(df,xticklabels=True, yticklabels=True,cmap='Greys',cbar = False)  ### Reds, Greys,Blues,PuRd

plt.xticks(rotation=0)
for ind, label in enumerate(s.get_xticklabels()):
    if ind % 16 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)    
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=False,left=True) 
plt.locator_params(axis='x', nbins=200)  

#plt.tick_params(axis=u'both',length=15,color='black',bottom=True,left=True)                                               
plt.tight_layout() 

##########################################  DD HMAP
      

############   24den sonraki hal--- burayi duzenle-- 24 yoksa bir oncekini alsin ama 24 varsa oradan itibaren alsin, cunku data 24den baslayacak

# datDD_t[0]['x'] = datDD_t[0]['x'][4:23]
# datDD_t[0]['y'][0] = datDD_t[0]['y'][0][4:23]
# datDD_t[0]['y'][1] = datDD_t[0]['y'][1][4:23]

# datDD_t[1]['x'] = datDD_t[1]['x'][4:23]
# datDD_t[1]['y'][0] = datDD_t[1]['y'][0][4:23]
# datDD_t[1]['y'][1] = datDD_t[1]['y'][1][4:23]

###################################################  altta fnc yazilmisi var

### alttakini fonk olarak yaz    ------ data 1-data 2 birlestirici fonk yaz !  hem LD icin hem DD icin !
solution = sol_DD_t
data_threshold = datDD_t
data1={}
#data1['x'] = data1['x'] + 3
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[0]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[0]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[0]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[0]['y'][1][j-1])
            y = [y0,y1]               
            break
data1['x'] = time
data1['y']= y

### alttakini fonk olarak yaz
data2={}
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[1]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[1]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[1]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[1]['y'][1][j-1])
            y = [y0,y1]               
            break
data2['x'] = time
data2['y']= y

#dark_cbar = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
##############  this is for LD colormap
time_list = np.linspace(24,120,193)

def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0 = []
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data['x']):         
            if val<=value:
                y0.append(aggregated_data['y'][j-1])
                y = y0             
                break                 
    while len(time_list)!= len(y0):
        y.append(aggregated_data['y'][-1])
    return y  

light_switch= []
for i in range(4,21) :
    if i == 4:
        light_switch.append(6*i)    
    if i % 2 == 1:
        light_switch.append(6*i)    
light_cbar = {}
light_cbar['x'] = light_switch
light_cbar['y'] = [1,1,1,1,1,1,1,1,1] #[1,0,1,0,1,0,1,0,1]  
T = time_equalise(time_list,light_cbar,m=0)
##################


def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0,y1 = [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(aggregated_data[m]['y'][0][j-1])
                y1.append(aggregated_data[m]['y'][1][j-1])
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(aggregated_data[m]['y'][0][-1])
        y[1].append(aggregated_data[m]['y'][1][-1])  
    return y      
cb = []
for i in range(len(T)):
    cb.append(float(T[i]))
    



aggregated_data = datDD_t   #######################################! change
first_data = time_equalise(time_list,aggregated_data,0)   
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_DD_t                    #######################################! change
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)],dtype=object)
harvest[0] = cb
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]

y_axis_labels = ['Light regime(LD/DD)','mRNA data', 'mRNA prediction', 'mRNA data', 'mRNA prediction','Protein data', 'Protein prediction','Protein data', 'Protein prediction'] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,  # np.linspace(24, 120, 26)
                   index = y_axis_labels,dtype="float")


plt.figure()
matplotlib.rc('figure', figsize=(20, 9))
sns.set(font_scale=1.9)
s = sns.heatmap(df,xticklabels=True, yticklabels=True,cmap='Greys',cbar = False)  ### Reds, Greys,Blues,PuRd

plt.xticks(rotation=0)
for ind, label in enumerate(s.get_xticklabels()):
    if ind % 16 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)    
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=False,left=True) 
plt.locator_params(axis='x', nbins=200)  

#plt.tick_params(axis=u'both',length=15,color='black',bottom=True,left=True)                                               
plt.tight_layout() 













