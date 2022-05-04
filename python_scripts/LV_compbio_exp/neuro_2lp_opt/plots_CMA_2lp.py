#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 23:54:50 2022

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
#####    Readings and the plots

#%%
#####   CMA-ES


#############################################  CMA-ES MI 

with open("Desktop/Llyonesse/Neuro_2lp_res/CMA_MI_neuro2lp/x_CMAES_MI_neuro2lp.txt", "rb") as fp:   
    x_CMA_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/Neuro_2lp_res/CMA_MI_neuro2lp/f_CMAES_MI_neuro2lp.txt", "rb") as fp:   
    f_CMA_MI = pickle.load(fp)  
    
#############################################  CMA-ES GG

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(32):
    gate = gatesm[k]
    savename = f"{gate}"    
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/CMA_GG_neuro2lp/x_CMAES_{savename}.txt", "rb") as fp:   
        globals()['x_CMAES_%s' % gate] = pickle.load(fp) 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/CMA_GG_neuro2lp/f_CMAES_{savename}.txt", "rb") as fp:   
        globals()['f_CMAES_%s' % gate]= pickle.load(fp) 
    globals()['x_CMAES_%s' % gate] = globals()['x_CMAES_%s' % gate][0:30]
    globals()['f_CMAES_%s' % gate] = globals()['f_CMAES_%s' % gate][0:30]
    

#%%
##########################################################   Bar chart- CMA-ES - MI    

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
bar[0].set_color('purple')
plt.title('CMA-ES: MI',fontsize=25)
plt.savefig('Desktop/Figs_neuro2lp/CMA_ES_MI_frequency.eps', format='eps',bbox_inches='tight')



##########################################################   Box plot- CMA-ES - gxG   
#data = []
xlabs = []
for k in range(32):
    xlabs.append(str(k))
    #data.append(globals()['f_g%s' % k])



data = []
for k in range(32):
    gate = gatesm[k]
    globals()['f_g%s' % k] = globals()['f_CMAES_%s' % gate] 
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
plt.title("CMA-ES:GxG",fontsize=20)
plt.savefig('Desktop/Figs_neuro2lp/CMA_GG_P_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()

####################################################################################   PCP for gate 6 -   both CMAES and NMMSO [0,0,1,1,0] 

c1=[]
c2=[]

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[6]
savename = f"{gate}"

gate_1g = globals()[f'x_%s_%s_%s' % (model,f"neuro2lp",gate)]
f = globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)]

f = []

for i in globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)]:
    f.append(i)
for i in globals()[f'f_%s_%s_%s' % ("NMMSO",f"neuro2lp",gate)]:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0,6) #(min(f), max(f)) ...   #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    
for i in range(len(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)])):        
    if globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i])]:
        c1.append(colours[f.index(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i])])   
for i in range(len(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)])):        
    if globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i])]:
        c2.append(colours[f.index(globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)][i])])   



###  plot  CMAES

model = "CMAES"   # "NMMSO"# 


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
for i in globals()[f'x_%s_%s_%s' % (model,f"neuro2lp",gate)]:
    print(i)
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': globals()[f'f_%s_%s_%s' % (model,f"neuro2lp",gate)]} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$T_1$','$T_2$','$T_3$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,6) #(min(f), max(f))  !! color...
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
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=6))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
#plt.savefig('Desktop/Figs_neuro2lp/CMAES_GG_6_pcp_naxes.eps', format='eps',bbox_inches='tight')
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
plt.savefig('Desktop/Figs_neuro2lp/NMMSO_GG_6_pcp_naxes.eps', format='eps',bbox_inches='tight')
plt.show()    





####################################################################################   PCP for gate 7 -   both CMAES and NMMSO [0,0,1,1,1] 

c1=[]
c2=[]

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[7]
savename = f"{gate}"

gate_1g = globals()['x_CMAES_%s' % savename]
f = globals()['f_CMAES_%s' % savename]

f = []

for i in globals()['f_CMAES_%s' % savename]:
    f.append(i)
for i in globals()['f_NMMSO_%s' % savename]:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0, 6) #(min(f), max(f))    #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    
for i in range(len(globals()['f_CMAES_%s' % savename])):        
    if globals()['f_CMAES_%s' % savename][i]==f[f.index(globals()['f_CMAES_%s' % savename][i])]:
        c1.append(colours[f.index(globals()['f_CMAES_%s' % savename][i])])   
for i in range(len(globals()['f_NMMSO_%s' % savename])):        
    if globals()['f_NMMSO_%s' % savename][i]==f[f.index(globals()['f_NMMSO_%s' % savename][i])]:
        c2.append(colours[f.index(globals()['f_NMMSO_%s' % savename][i])])   



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








 














#####################################################################################  COMPARABLE SOLS


####################################################################################   PCP for gate 7 -   both CMAES and NMMSO [0,0,1,1,1] 

c1=[]
c2=[]

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[7]
savename = f"{gate}"

gate_1g = globals()['x_CMAES_%s' % savename]
gate_1 =[] 
gate_1.append(gate_1g[6]) 
gate_1.append(gate_1g[11])
gate_1g = gate_1
f = globals()['f_CMAES_%s' % savename]

fn = []
fn.append(f[6])
fn.append(f[11])
f = fn


f = sorted(f)
norm = colors.Normalize(0, 6) #(min(f), max(f))    #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    
for i in range(len(globals()['f_CMAES_%s' % savename])):        
    if globals()['f_CMAES_%s' % savename][i]==f[f.index(globals()['f_CMAES_%s' % savename][i])]:
        c1.append(colours[f.index(globals()['f_CMAES_%s' % savename][i])])   
for i in range(len(globals()['f_NMMSO_%s' % savename])):        
    if globals()['f_NMMSO_%s' % savename][i]==f[f.index(globals()['f_NMMSO_%s' % savename][i])]:
        c2.append(colours[f.index(globals()['f_NMMSO_%s' % savename][i])])   



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



#######################   Threshold histogram



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
bar[0].set_color('purple')
plt.title('CMA-ES: MI',fontsize=25)
plt.savefig('Desktop/Figs_neuro2lp/CMA_ES_MI_frequency.eps', format='eps',bbox_inches='tight')



##############################################################################################   extract sols final !  gate= 6 CMA

c1=[]
c2=[]

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[6]
savename = f"{gate}"
gate_1g = globals()['x_CMAES_%s' % savename]
gate_1 =[] 
gate_1.append(gate_1g[0]) 
gate_1.append(gate_1g[21])
f = globals()['f_CMAES_%s' % savename]

fn = []
fn.append(f[0])
fn.append(f[21])



f = sorted(f)
norm = colors.Normalize(0, 6) #(min(f), max(f))    #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    




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
for i in gate_1:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    T1.append(i[5])
    T2.append(i[6])
    T3.append(i[7])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,'Fitness': fn} 
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
        im = ax.plot(df.loc[idx, cols], color= colours[i])
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
plt.savefig('Desktop/CMAES_GG_6_pcp_naxes.eps', format='eps',bbox_inches='tight')
plt.show()    














##############################  pairwise CMA g6 00110

###############################################################################     

    """
    1) Combine x and f values in a list of zips- this gives list of tuples 
    2) sort this combined list based on 1st index where f is
    3) Have 2 lists of x list and flist to use
    4) Create binary combination of all items in x including the sol itself
    5) Linearly space 10 equal points in each coordinate, here ndim = 5, each coordinate in each line divided by 10, have list l and append this(size of 5 each having 10 equal-to another list inside)
    6) f_pairs is the list of all f values with length of binary combinations having 10 f in each
    7) corr_x gives f like (0,0),(0,1),...(0,30),(1,1),(1,2)...)))) lower or upper triangle only one side
    8) f_range is 465 differences between all binary combined x values
    9) colours1 are the normalised colours within  f_range ---colours2 are normalised colours for f values 30 different (c1 measures the diff(465) c2 measures each sol quality(30))
    10) f_col seperates colors1 as to be used in pairwise plot 
    11)
    """
n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[6]
savename = f"{gate}"


#for i in range(len(globals()['f_CMAES_%s' % gate])):
    #globals()['f_CMAES_%s' % gate][i] = -1 * globals()['f_CMAES_%s' % gate][i]


gate_1g = globals()['x_CMAES_%s'% savename]
fit = globals()['f_CMAES_%s' % savename]



comb = list(zip(globals()['x_CMAES_%s' % savename],fit))
sorted_list = sorted(comb, key=lambda x: x[1])
xlist = []
flist_cma =[]
for i in sorted_list:
    xlist.append(i[0])
    flist_cma.append(i[1])

# obtain combinations of two of all the solutions
comb_two = []
for i in range(len(xlist)):
    for j in range(i,len(xlist)):  #simply start from i+1 not to include solution itself
        comb_two.append([xlist[i],xlist[j]])

# linearly spaced 10 points between all combinations of two of solutions in each dimension (5) (since last 2 are all the same in this case)
n_dim =8       
sampled_pairs = []  
for i in range(len(comb_two)):
    l = []
    for j in range(n_dim):
        l.append(list(np.linspace(comb_two[i][0][j], comb_two[i][1][j], 10)))
    sampled_pairs.append(l)
    
# for fitness-- go NMMSO GxG

def fitness(inputparams,gates=[0,0,1,1,0]):
  inputparams = list(inputparams)
  inputparams = matlab.double(inputparams)
  gates = matlab.double(gates)
  cost = eng.getBoolCost_neuro2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
  return cost


# all fitness for sampled points inside f_pairs
f_pairs =[] 
tot_input= []
for i in range(len(sampled_pairs)):  # 1,5 min with length of 450
    start_time = time.time()
    l_loc =[]
    k = sampled_pairs[i]
    for t in range(10):
        inputparams =[]  
        for n in range(8):    ##  !!!!   each coordinate yani bunu degistir
            inputparams.append(float(k[n][t]))
        tot_input.append(inputparams)
        l_loc.append(fitness(inputparams,gates=[0,0,1,1,0]))    
    f_pairs.append(l_loc)
    print("--- %s seconds ---" % (time.time() - start_time))   

with open("Desktop/f_pairs_CMA_2lp_6.txt", "wb") as fp:   
    pickle.dump(f_pairs, fp)  
with open("Desktop/f_pairs_CMA_2lp_6.txt", "rb") as fp:   
    f_pairs = pickle.load(fp)  



######################################
# fitness list of lists associated with the solution idx 
n = 30
corr_x = []
c = 30
k = 0
for i in range(n):  
    corr_x.append(f_pairs[k:k+c])
    k = k+c
    c = c-1
 


########################### pairwise plot :
 
f_range = []
for i in range(len(f_pairs)):
 f_range.append(f_pairs[i][9]-f_pairs[i][0])         

norm = colors.Normalize(min(f), max(f))
colours = cm.nipy_spectral(norm(f_range))  # div  RdYlBu gnuplot gist_ncar  nipy_spectral hsv



labels = []
for i in range(0,30):      
    labels.append(str(i+1))
labelsy = labels 
labelsy.reverse()
labels = []
for i in range(0,30):      
    labels.append(str(i+1))


n = 30
f_col = []
c = 30
k = 0
for i in range(n):  
    f_col.append(colours[k:k+c])
    k = k+c
    c = c-1

n = 30
fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='white')   # ,facecolor='white',facecolor='lightyellow'...g, axes = plt.subplots(nrows=30, ncols=30,figsize=(10,10))

gs = fig.add_gridspec(nrows=n, ncols=n)
ax = np.zeros((n, n), dtype=object)
ax2 = plt.axes()
loc = range(len(labels))
labelsa = labels
plt.xticks(loc, labels, rotation='horizontal',fontsize=4)
plt.yticks(loc, labelsy, rotation='horizontal',fontsize=4)
#ax2 = plt.axes([0.15, 0.15, 0.9, 0.9])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
for i in range(n):
    for j in range(0+i,n):
        if j > i:
            ax[j,i] = fig.add_subplot(gs[j,i])
            ax[j,i].set_ylim([0,1])
            #ax[j,i].set_title('gs[0, :]')
            ax[j,i].plot(corr_x[i][j-i],linewidth=0.75,color=f_col[i][j-i])
            ax[j,i].axis('off') 
            
            ax[i,j] = fig.add_subplot(gs[i,j])
            ax[i,j].set_ylim([0.04,0.65])
            #ax[j,i].set_title('gs[0, :]')
            ax[i,j].plot(corr_x[i][j-i],linewidth=0.75,color=f_col[i][j-i])
            ax[i,j].axis('off')       
        if j == i:
             ax[j,i] = fig.add_subplot(gs[j,i])  
             ax[j,i].plot((0.5), (0.5), 'o', color=c2[i])
             ax[j,i].axis('off')

                 
             
plt.axis('off')
sm = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm2)
position=fig.add_axes([0.97,0.125,0.03,0.75])
#position=fig.add_axes([0.91,0.3,0.03,0.49]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
a = plt.colorbar(sm,cax=position,pad = 0.8,shrink=0.4)# pad = 0.15 shrink=0.9 colorbar length
a.update_ticks()
a.ax.tick_params(labelsize=6)
#sm2 = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm2)
#position2 =fig.add_axes([1.1,0.05,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
#a2 = plt.colorbar(sm2,cax=position2,shrink=0.9)# pad = 0.15 shrink=0.9 colorbar length
#a2.ax.tick_params(labelsize=6) 
plt.savefig('Desktop/pairwise_2lp_cmag6_.eps', format='eps',bbox_inches='tight')
plt.show()


######################################################################################################  pairwise end






