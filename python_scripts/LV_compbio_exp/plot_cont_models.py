#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:18:55 2022

@author: melikedila
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
import seaborn as sns
import pandas as pd
import plotly.express as px
import pandas
from pandas.plotting import parallel_coordinates
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import statistics as st
import itertools
import matlab
from matplotlib import ticker
from colour import Color
import matplotlib 
import matplotlib.cm as cmx

#%%
#####    Readings and the plots

###      NMMSO  ---- CMA 

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
                    

##############################################   Bar chart 1st models
##### make sure last gates are rounded for NMMSO

for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):
    i[5] = round(i[5])
    i[6] = round(i[6])
    #print(i)
### now plot    

gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []


opt = "NMMSO" # "CMAES" #
model = "neuro1lp" #,"neuro2lp": 5}

for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(globals()['f_%s_MI_%s' % (opt,model)][j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(globals()['f_%s_MI_%s' % (opt,model)][j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(globals()['f_%s_MI_%s' % (opt,model)][j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(globals()['f_%s_MI_%s' % (opt,model)][j])


x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('LC',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(0, 31, 2.0),fontsize=11)
plt.xticks(fontsize=20)
bar = plt.bar(x, height=y,color= 'royalblue')
bar[0].set_color('purple')
plt.title('NMMSO: MI',fontsize=25)
plt.savefig('Desktop/%s_MI_frequency_%s.eps'%(opt,model), format='eps',bbox_inches='tight')
#################################  Bar chart 2nd model


for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):   ### round first all NMMSO
    for t in range(8,13):
        i[t] = round(i[t])





n_gates = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
opt =  "CMAES" #"NMMSO" # 
model = "neuro2lp" #,"neuro2lp": 5}

for k in range(2**n_gates):
    gate = gatesm[k]
    #print(gate)
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):
        if i[8:13] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(globals()['x_%s_MI_%s' % (opt,model)][j])
            
x  = []  
y = []   
for k in range(2**n_gates): 
   x.append(str(k))  
   y.append(len(globals()['gate_%s'% k]))
            
merge = np.column_stack((x, y))
merge = merge[np.argsort(merge[:, 1])]
x = merge[:,0]
y = list(merge[:,1])

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
#bar[8].set_color('purple')
plt.title('%s: MI'%opt,fontsize=25)
plt.savefig('Desktop/CMA-ES_MI_frequency.eps', format='eps',bbox_inches='tight')

#################################  Bar chart 3rd model


for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):   ### round first all NMMSO
    for t in range(15,23):
        i[t] = round(i[t])


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []




n_gates = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  
opt =  "NMMSO" # "CMAES" #
model = "arabid2lp" #,"neuro2lp": 5}

for k in range(2**n_gates):
    gate = gatesm[k]
    #print(gate)
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(globals()['x_%s_MI_%s' % (opt,model)]):
        #print(i)
        if i[15:23] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(globals()['x_%s_MI_%s' % (opt,model)][j])
            
x  = []  
y = []   
for k in range(2**n_gates): 
   x.append(str(k))  
   y.append(len(globals()['gate_%s'% k]))
            
merge = np.column_stack((x, y))
merge = merge[np.argsort(merge[:, 1])]
x = merge[:,0]
y = list(merge[:,1])

X = []
for i in x:
    X.append(str(i))   
Y = []
for i in y:
    Y.append(int(i))     
y = Y

x_list,y_list = [], []
for i in range(2**n_gates):
    x_list.append(x[2**n_gates-1-i])
    y_list.append(y[2**n_gates-1-i])
    
x_list = x_list[0:16]
y_list = y_list[0:16]

fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=12,rotation=90)
bar = plt.bar(x_list, height=y_list,color= 'royalblue')
#bar[1].set_color('purple')
#bar[8].set_color('purple')
plt.title('%s: MI'%opt,fontsize=25)
plt.savefig('Desktop/NMMSO_MI_frequency_%s.eps'%model, format='eps',bbox_inches='tight')






###################################################   boxplot --GG  1st mdoel
y_llimit = 0
y_ulimit = 4.2
    
n_gates = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  


xlabs = []
for k in range(2**n_gates):
    xlabs.append(str(k))
    #data.append(globals()['f_g%s' % k])

opt = "NMMSO" #"CMA-ES" #
model = "neuro1lp" #,"neuro2lp": 5}


data = []
for k in range(2**n_gates):
    gate = gatesm[k]
    globals()['f_g%s' % k] = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
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
plt.ylim((y_llimit,y_ulimit))
plt.title("%s:GxG"%opt,fontsize=20)
plt.arrow(x=1, y=2, dx=0, dy=-1, width=.03, facecolor='red', edgecolor='none')
plt.savefig('Desktop/%s_GG_boxplt_%s.eps'%(opt,model), format='eps',bbox_inches='tight')
plt.show()

#################################   2nd model boxplot

opt = "NMMSO" # "CMAES" #
model = "neuro2lp" #,"neuro2lp": 5}
n_gates = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  

# from negative to positive for NMMSO
# for k in range(32):
#     gate = gatesm[k]
#     for i in range(len(globals()[f'f_%s_%s_%s' % (opt,model,gate)])):
#         globals()[f'f_%s_%s_%s' % (opt,model,gate)][i]= -1*globals()[f'f_%s_%s_%s' % (opt,model,gate)][i]

xlabs = []
for k in range(32):
    xlabs.append(str(k))
    #data.append(globals()['f_g%s' % k])



data = []
for k in range(32):
    gate = gatesm[k]
    globals()['f_g%s' % k] = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
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
plt.title("%s:GxG"%opt,fontsize=20)
plt.arrow(x=2, y=3.5, dx=0, dy=-1, width=.15, facecolor='red', edgecolor='none')
plt.arrow(x=3, y=3.5, dx=0, dy=-1, width=.16, facecolor='red', edgecolor='none')
plt.savefig('Desktop/%s_GG_boxplt_neuro2lp.eps'%opt, format='eps',bbox_inches='tight')
plt.show()

#################################################################################################   PCP 1st model  for NMMSO CMAES

#########################################################################    GG 01
opt = "NMMSO" # "CMAES" #
model = "neuro1lp" #,"neuro2lp": 5}
n_gates =  model_list["neuro1lp"]
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  
gate = gatesm [1]

#######  NMMSO gxg data

fn = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
xn = globals()[f'x_%s_%s_%s' % (opt,model,gate)]

########  CMA-ES gxg data
opt = "CMAES" #"NMMSO" # 

xc = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
fc = globals()[f'f_%s_%s_%s' % (opt,model,gate)]




###########################################  buu  NMMSO
### cloorlist
f = []

for i in fn:
    f.append(i)
for i in fc:
    f.append(i)
    
    
f = sorted(f)
norm = colors.Normalize(0, 4)
colours = cm.RdBu(norm(f))  # div

c1 = []
c2 = []
   
for i in range(len(fn)):        
    if fn[i]==f[f.index(fn[i])]:
        c1.append(colours[f.index(fn[i])])   
for i in range(len(fc)):        
    if fc[i]==f[f.index(fc[i])]:
        c2.append(colours[f.index(fc[i])])   

###  plot
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12,'$T_1$' :1,'$T_2$' : 1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0,'$T_1$' :0,'$T_2$' : 0}


t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]
for i in xn:              # !!!
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': fn} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,4)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c1[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=4))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/NMMSO_GG_PCP_neuro1lp_1.eps', format='eps',bbox_inches='tight')
plt.show()   
        
#########################################################################################     CMA

t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]
for i in xc:              # !!!
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])


data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': fc} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,4)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c2[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=4))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/CMA_GG_PCP_neuro1lp_1.eps', format='eps',bbox_inches='tight')
plt.show()   
    
##########################################################################################################  2nd model   gates 15-6-7 PCPs
##################################################################################################################



#########################################################################    
opt = "NMMSO" # "CMAES" #
model = "neuro2lp" #,"neuro2lp": 5}
n_gates =  model_list["neuro2lp"]
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  
gate = gatesm [1]

#########################################################################    GG 15
#######  NMMSO gxg data
gate = gatesm [15]

fnfif = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
xnfif = globals()[f'x_%s_%s_%s' % (opt,model,gate)]

########  CMA-ES gxg data
opt = "CMAES" #"NMMSO" # 

xcfif = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
fcfif = globals()[f'f_%s_%s_%s' % (opt,model,gate)]

#########################################################################    GG 6
#######  NMMSO gxg data
gate = gatesm [6]
opt = "NMMSO"

fnsix = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
xnsix = globals()[f'x_%s_%s_%s' % (opt,model,gate)]

########  CMA-ES gxg data
opt = "CMAES" #"NMMSO" # 

xcsix = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
fcsix = globals()[f'f_%s_%s_%s' % (opt,model,gate)]

#########################################################################    GG 7
#######  NMMSO gxg data
gate = gatesm [7]
opt = "NMMSO"

fnsev = globals()[f'f_%s_%s_%s' % (opt,model,gate)]
xnsev = globals()[f'x_%s_%s_%s' % (opt,model,gate)]

########  CMA-ES gxg data
opt = "CMAES" #"NMMSO" # 

xcsev = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
fcsev = globals()[f'f_%s_%s_%s' % (opt,model,gate)]


### cloorlist
f = []

for i in fnfif:
    f.append(i)
for i in fcfif:
    f.append(i)
for i in fnsix:
    f.append(i)
for i in fcsix:
    f.append(i)
for i in fnsev:
    f.append(i)
for i in fcsev:
    f.append(i)
    
    
f = sorted(f)
norm = colors.Normalize(0, 6)
colours = cm.RdBu(norm(f))  # div

c1,c2,c3,c4,c5,c6 = [[] for i in range(6)]

   
for i in range(len(fnfif)):        
    if fnfif[i]==f[f.index(fnfif[i])]:
        c1.append(colours[f.index(fnfif[i])])   
for i in range(len(fcfif)):        
    if fcfif[i]==f[f.index(fcfif[i])]:
        c2.append(colours[f.index(fcfif[i])])   
for i in range(len(fnsix)):        
    if fnsix[i]==f[f.index(fnsix[i])]:
        c3.append(colours[f.index(fnsix[i])])   
for i in range(len(fcsix)):        
    if fcsix[i]==f[f.index(fcsix[i])]:
        c4.append(colours[f.index(fcsix[i])])   
for i in range(len(fnsev)):        
    if fnsev[i]==f[f.index(fnsev[i])]:
        c5.append(colours[f.index(fnsev[i])])   
for i in range(len(fcsev)):        
    if fcsev[i]==f[f.index(fcsev[i])]:
        c6.append(colours[f.index(fcsev[i])])  
         
###########################################  buu  NMMSO    GG 15 NMMSO
###  plot
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12, '$τ_4$':24, '$τ_5$':12,'$T_1$' :1,'$T_2$' : 1,'$T_3$' :1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$T_1$' :0,'$T_2$' : 0, '$T_3$' :0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xnfif:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fnfif} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c1[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/NMMSO_G15_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')
plt.show()   
        
#########################################################################################     CMA
t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xcfif:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fcfif} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c2[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/CMA_G15_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')
plt.show()   


###########################################  buu  NMMSO    GG 6 NMMSO
###  plot
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12, '$τ_4$':24, '$τ_5$':12,'$T_1$' :1,'$T_2$' : 1,'$T_3$' :1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$T_1$' :0,'$T_2$' : 0, '$T_3$' :0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xnsix:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fnsix} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c3[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/NMMSO_G6_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')
plt.show()   
        
#########################################################################################     CMA
t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xcsix:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fcsix} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c4[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/CMA_G6_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')
    


###########################################  buu  NMMSO    GG 7 NMMSO
###  plot
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12, '$τ_4$':24, '$τ_5$':12,'$T_1$' :1,'$T_2$' : 1,'$T_3$' :1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$T_1$' :0,'$T_2$' : 0, '$T_3$' :0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xnsev:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fnsev} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c5[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/NMMSO_G7_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')
plt.show()   
        
#########################################################################################     CMA
t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]

for i in xcsev:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fcsev} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c6[i])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/CMA_G7_PCP_neuro2lp.eps', format='eps',bbox_inches='tight')

###########################################################################################  Extract solutions for the first model

####################################################################################   PCP for gate 1 -   both CMAES and NMMSO [
opt = "CMAES" #"NMMSO" # 
model = "neuro1lp" #,"neuro2lp": 5}
n_gates = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))  
gate = gatesm[1]
savename = f"{gate}"



c1=[]
c2=[]



gate_1g = globals()[f'x_%s_%s_%s' % (opt,model,gate)]
gate_1 =[] 
gate_1.append(gate_1g[12]) 
gate_1.append(gate_1g[11])
gate_1g = gate_1
f = globals()[f'f_%s_%s_%s' % (opt,model,gate)]

fn = []
fn.append(f[6])
fn.append(f[11])
f = fn


f = sorted(f)
norm = colors.Normalize(0, 4) #(min(f), max(f))    #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
opt = "CMAES" #"NMMSO" #     
for i in range(len(globals()[f'f_%s_%s_%s' % (opt,model,gate)])):        
    if globals()[f'f_%s_%s_%s'%(opt,model,gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (opt,model,gate)][i])]:
        c1.append(colours[f.index(globals()[f'f_%s_%s_%s' % (opt,model,gate)][i])])   

opt = "NMMSO" #   "CMAES" #      
for i in range(len(globals()[f'f_%s_%s_%s' % (opt,model,gate)])):        
    if globals()[f'f_%s_%s_%s' % (opt,model,gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (opt,model,gate)][i])]:
        c2.append(colours[f.index(globals()[f'f_%s_%s_%s' % (opt,model,gate)][i])])   



###  plot  CMAES
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12, '$τ_4$':24, '$τ_5$':12,'$T_1$' :1,'$T_2$' : 1,'$T_3$' :1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$T_1$' :0,'$T_2$' : 0, '$T_3$' :0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]
for i in globals()['x_CMAES_%s' % savename]:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': globals()['f_CMAES_%s' % savename]} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c1[idx])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figs_neuro2lp/CMAES_GG_7_pcp_naxes.eps', format='eps',bbox_inches='tight')
plt.show()    



###       plot   NMMSO



T1 =[]
T2 =[]
T3 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]
for i in globals()['x_NMMSO_%s' % savename]:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': globals()['f_NMMSO_%s' % savename]} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0, 6)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]]))
    #df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c2[idx])
    ax.set_xlim([x[i], x[i+1]])
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    tick_labels = [round(min_val + step * i, 1) for i in range(ticks)]
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=7) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=7)
ax.set_xticklabels([cols[-2], cols[-1]])
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figs_neuro2lp/NMMSO_GG_7_pcp_naxes.eps', format='eps',bbox_inches='tight')
plt.show()  






































    