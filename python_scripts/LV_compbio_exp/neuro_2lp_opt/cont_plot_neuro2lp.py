#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:06:33 2022

@author: melikedila
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
#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro2lp_costfcn"
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
#####    Readings and the plots

#############################################  CMA MI

with open("Desktop/Llyonesse/Neuro_2lp_res/cont/x_CMAES_MI_cts.txt", "rb") as fp:   
    x_CMA_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/Neuro_2lp_res/cont/f_CMA_MI_cont.txt", "rb") as fp:   
    f_CMA_MI = pickle.load(fp)  

#############################################  NMMSO MI

with open("Desktop/Llyonesse/Neuro_2lp_res/cont/design_dict_NMMSO_MI_cts_neuro2lp.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/Neuro_2lp_res/cont/fit_dict_NMMSO_MI_cts_neuro2lp.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/Neuro_2lp_res/cont/f_NMMSO_MI_ctsneuro2lp.txt", "rb") as fp:   
    f_NMMSO_MI = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_2lp_res/cont/x_NMMSO_MI_cts_neuro2lp.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)       
   
#############################################  CMA-ES GG

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(32):
    gate = gatesm[k]
    savename = f"{gate}"    
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/x_CMAES_{savename}_cts.txt", "rb") as fp:   
        globals()['x_CMAES_%s' % gate] = pickle.load(fp) 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/cont/f_CMAES_{savename}_cts.txt", "rb") as fp:   
        globals()['f_CMAES_%s' % gate]= pickle.load(fp) 
    globals()['x_CMAES_%s' % gate] = globals()['x_CMAES_%s' % gate][0:30]
    globals()['f_CMAES_%s' % gate] = globals()['f_CMAES_%s' % gate][0:30]
  
#############################################  NMMSO GG

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(32):
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





########################      barchart
for k in range(32):
    gate = gatesm[k]
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(x_CMA_MI):
        if i[8:19] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(f_CMA_MI[j])
            
x  = []  
y = []   
for k in range(32): 
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

##################################   NMMSO GG boxplot


#data = []
xlabs = []
for k in range(32):
    xlabs.append(str(k))
    #data.append(globals()['f_g%s' % k])

for k in range(32):
    gate = gatesm[k]
    for i in range(len(globals()['f_NMMSO_%s' % gate])):
        globals()['f_NMMSO_%s' % gate][i] = -1 * globals()['f_NMMSO_%s' % gate][i]

    

data = []
for k in range(32):
    gate = gatesm[k]
    globals()['f_g%s' % k] = globals()['f_NMMSO_%s' % gate] 
    data.append(globals()['f_g%s' % k])
    
f_median = []
for i in data:
    f_median.append(np.median(i))    
    

comb = list(zip(data,f_median))
sorted_list = sorted(comb, key=lambda x: x[1])
xdata = []
fmedian =[]
for i in sorted_list:
    xdata.append(i[0])
    fmedian.append(i[1])

comb = list(zip(xlabs,f_median))
sorted_list = sorted(comb, key=lambda x: x[1])
xlist = []
fmedian =[]
for i in sorted_list:
    xlist.append(i[0])
    fmedian.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(xdata,notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(xlist)
plt.yticks(fontsize=15)
plt.xticks(fontsize=12,rotation=90)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,6.2))
plt.title("NMMSO:GxG",fontsize=20)
plt.savefig('Desktop/cont_figs/NMMSO_GG_2lp_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()


