#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:01:16 2022

@author: mkaratas
"""

import matlab.engine
import numpy as np
import pickle
import matplotlib.pyplot as plt
#from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
import seaborn as sns
import pandas as pd
import plotly.express as px
import pandas
from pandas.plotting import parallel_coordinates
import matplotlib as mpl
import statistics as st
from matplotlib.cm import ScalarMappable
import itertools
import matplotlib 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.cm as cm
from matplotlib import ticker
from colour import Color

eng = matlab.engine.start_matlab()

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



n_gates = 8
gatesm = list(map(list, itertools.product([0,1], repeat=n_gates)))   


class Mat_Py:
    def __init__(self):
        self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd = ([] for _ in range(5))
        
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


opt = "CMAES" #"NMMSO" # 
model = "arabid2lp"
gate = gatesm[153]
for idx,sol in enumerate(globals()[f'x_%s_%s_%s' % (opt,model,gate)]):  
    print(sol)
    gates = list(gate)
    gates = matlab.double(gates)
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    print(inputparams)
    func_output_CMA.update(*eng.getBoolCost_cts_arabid2lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))

[cost_CMA, sol_LD_CMA, sol_DD_CMA, datLD_CMA, datDD_CMA] = func_output_CMA.get_all()
# print(cost_values.sol_dd[0])
# list(cost_values.sol_dd[4]['x']._data)
# cost_values.sol_dd[4]['x'][0]._data
# list(cost_values.sol_dd[4]['x'][0]._data)
# print(cost_values.get_all())


####  NMMSO


opt = "NMMSO"
gates = gatesm[153]
for idx,sol in enumerate(globals()[f'x_%s_%s_%s' % (opt,model,gate)]): 
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_NMMSO.update(*eng.getBoolCost_cts_arabid2lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_NMMSO, sol_LD_NMMSO, sol_DD_NMMSO, datLD_NMMSO, datDD_NMMSO] = func_output_NMMSO.get_all()    


##############

################################### normalisation--check
normalize= []  #336  #####   have all data appended to normalize list,, then append to no_dataLD_arr
                                ### here dataLD_arr in each row has a diffferent components, therefore we added 2nd row
for i in dataLD_arr[0]:
    normalize.append(i)
for i in dataLD_arr[1]:
    normalize.append(i)  
for i in dataLD_arr[2]:
    normalize.append(i)  
for i in dataLD_arr[3]:
    normalize.append(i)      
for i in dataDD_arr[0]:
    normalize.append(i)
for i in dataDD_arr[1]:
    normalize.append(i) 
for i in dataDD_arr[2]:
    normalize.append(i)  
for i in dataDD_arr[3]:
    normalize.append(i)      
N = NormalizeData(normalize)

no_dataLD_arr = []
no_dataDD_arr = []
n_dark = 251
n_light = 241
no_dataLD_arr.append(N[0:n_light])
no_dataLD_arr.append(N[n_light:2*n_light])  ## 241 data sayisi
no_dataLD_arr.append(N[2*n_light:3*n_light]) 
no_dataLD_arr.append(N[3*n_light:4*n_light])   ## 251 data sayisi
no_dataDD_arr.append(N[4*n_light:4*n_light+n_dark])
no_dataDD_arr.append(N[4*n_light+n_dark:4*n_light+2*n_dark])
no_dataDD_arr.append(N[4*n_light+2*n_dark:4*n_light+3*n_dark])
no_dataDD_arr.append(N[4*n_light+3*n_dark:4*n_light+4*n_dark])

no_dataLD_arr[0] = no_dataLD_arr[0][48:]
no_dataLD_arr[1] = no_dataLD_arr[1][48:]
no_dataLD_arr[2] = no_dataLD_arr[2][48:]
no_dataLD_arr[3] = no_dataLD_arr[3][48:]
no_dataDD_arr[0] = no_dataDD_arr[0][48:]
no_dataDD_arr[1] = no_dataDD_arr[1][48:]
no_dataDD_arr[2] = no_dataDD_arr[2][48:]
no_dataDD_arr[3] = no_dataDD_arr[3][48:]
no_dataLD_arr.append(dataLD_arr[-1][48:])
no_dataDD_arr.append(dataDD_arr[-1][48:])


##########
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

####  Unit test for partition  --- su an calismiyor kalsin

a = [[1,2,3,4,5,6,7,8],[0,0,1,1,1,0,1,0]]
b = [[2,3,8]]

sl = partition(a, b)

#####  Time Series for the 3rd Model

######################################################################  Data TS LD

df = {'time': no_dataLD_arr[4],
  'A':no_dataLD_arr[0],
  'B':no_dataLD_arr[1],
  'C':no_dataLD_arr[2],
  'D':no_dataLD_arr[3]}    
dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list 12LD 12LL


####  bi ona bi ona light-dark saatler ayrilmis
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


matplotlib.rc('figure', figsize=(20, 7))
l1 = plt.plot(df['time'], df['A'],linewidth=4,label='$LHY$',color='lightsalmon')
l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$TOC_{1}$',color='rebeccapurple')
l3 = plt.plot(df['time'], df['C'],linewidth=4,label='$X$',color='blue')
l4 = plt.plot(df['time'], df['D'],linewidth=4,label='$Y$',color='red')
plt.xlabel('time (h)',fontsize=30)
plt.ylabel('Normalised expression levels',fontsize=30)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 27, alpha=1)
plt.yticks(np.arange(0, 1.1, 0.2),fontsize= 27, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.12),
          fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='gray')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_data_norm_arabid2lp.eps', format='eps',bbox_inches='tight')
plt.show()




#################################################################################################### CMA LD TS


matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][0] ,linewidth=2,label='$LHY$',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][1],linewidth=2,label='$TOC_{1}$',color='rebeccapurple')#sol_LD
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][2],linewidth=2,label='$X$',color='blue')#sol_LD
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][3],linewidth=2,label='$Y$',color='red')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][0] ,linewidth=2,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][1],linewidth=2,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][2],linewidth=2,color='blue')
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][3],linewidth=2,color='red')
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Expression (0/1)',fontsize=30)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_CMA_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()

##############  Line by line for each component
#####  LHY

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][0] ,linewidth=6,label='$LHY$',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable

for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][0] ,linewidth=6,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    
    
    
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
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_CMA_LHY_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()

#####  TOC1 line by line

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][1],linewidth=6,label='$TOC_{1}$',color='rebeccapurple')
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][1],linewidth=6,color='rebeccapurple')      #this indexfr the state of the first variable

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=1,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_CMA_TOC1_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()


#####  X line by line

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][2],linewidth=6,label='$X$',color='blue')
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][2],linewidth=6,color='blue')     #this indexfr the state of the first variable

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=1,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_CMA_X_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()


#####  Y line by line

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:4][3],linewidth=6,label='$Y$',color='red')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:4][3],linewidth=6,color='red')#sol_LD  #this indexfr the state of the first variable

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=1,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_CMA_Y_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()

##################################################################################################### 
##################################################################################################### 
##################################################################################################### NMMSO LD TS


matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:4][0] ,linewidth=1,label='$LHY$',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:4][1],linewidth=1,label='$TOC_{1}$',color='rebeccapurple')#sol_LD
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:4][2],linewidth=1,label='$X$',color='blue')#sol_LD
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:4][3],linewidth=1,label='$Y$',color='red')#sol_LD
for i in range(1,len(sol_LD_NMMSO)):
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:4][0] ,linewidth=1,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:4][1],linewidth=1,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:4][2],linewidth=1,color='blue')
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:4][3],linewidth=1,color='red')
    

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Expression (0/1)',fontsize=30)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LD_ts_NMMSO_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()

#########################################################################################################  Data TS DD
df = {'time': no_dataDD_arr[4],
  'A':no_dataDD_arr[0],
  'B':no_dataDD_arr[1],
  'C':no_dataDD_arr[2],
  'D':no_dataDD_arr[3]}    
dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list 12LD 12LL


####  bi ona bi ona light-dark saatler ayrilmis
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


matplotlib.rc('figure', figsize=(20, 7))
l1 = plt.plot(df['time'], df['A'],linewidth=4,label='$LHY$',color='lightsalmon')
l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$TOC_{1}$',color='rebeccapurple')
l3 = plt.plot(df['time'], df['C'],linewidth=4,label='$X$',color='blue')
l4 = plt.plot(df['time'], df['D'],linewidth=4,label='$Y$',color='red')
plt.xlabel('time (h)',fontsize=30)
plt.ylabel('Normalised expression levels',fontsize=30)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 27, alpha=1)
plt.yticks(np.arange(0, 1.1, 0.2),fontsize= 27, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.12),
          fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='white')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LL_ts_data_norm_arabid2lp.eps', format='eps',bbox_inches='tight')
plt.show()

#################################################################################################### CMA DD TS
matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:4][0] ,linewidth=2,label='$LHY$',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:4][1],linewidth=2,label='$TOC_{1}$',color='rebeccapurple')#sol_LD
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:4][2],linewidth=2,label='$X$',color='blue')#sol_LD
plt.plot(sol_DD_CMA[0]['x'], sol_DD_CMA[0]['y'][:4][3],linewidth=2,label='$Y$',color='red')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:4][0] ,linewidth=2,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:4][1],linewidth=2,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:4][2],linewidth=2,color='blue')
    plt.plot(sol_DD_CMA[i]['x'], sol_DD_CMA[i]['y'][:4][3],linewidth=2,color='red')
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Expression (0/1)',fontsize=30)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='white')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LL_ts_CMA_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()

############################################################################################################ NMMSO DD TS

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:4][0] ,linewidth=1,label='$LHY$',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:4][1],linewidth=1,label='$TOC_{1}$',color='rebeccapurple')#sol_LD
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:4][2],linewidth=1,label='$X$',color='blue')#sol_LD
plt.plot(sol_DD_NMMSO[0]['x'], sol_DD_NMMSO[0]['y'][:4][3],linewidth=1,label='$Y$',color='red')#sol_LD
for i in range(1,len(sol_LD_NMMSO)):
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:4][0] ,linewidth=1,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:4][1],linewidth=1,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:4][2],linewidth=1,color='blue')
    plt.plot(sol_DD_NMMSO[i]['x'], sol_DD_NMMSO[i]['y'][:4][3],linewidth=1,color='red')
    

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Expression (0/1)',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=4,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='white')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
plt.savefig('Desktop/cont_figs/LL_ts_NMMSO_arabid2lp.eps', format='eps',bbox_inches='tight')    
plt.show()




####################################################################################################################







################################################################################  HEATMAP



















