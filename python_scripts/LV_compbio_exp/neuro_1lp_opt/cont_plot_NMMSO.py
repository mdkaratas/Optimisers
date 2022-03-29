#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:00:14 2022

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
import matlab
#%%
#####    Readings and the plots

###      NMMSO

# without penalty

#############################################  NMMSO MI without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO_MI_cont.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO_MI_cont.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_MI_cont.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_MI_cont.txt", "rb") as fp:   
    f_NMMSO_MI  = pickle.load(fp)         
    
#############################################  NMMSO 01 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO1_cont.txt", "rb") as fp:   
    design_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO1_cont.txt", "rb") as fp:   
    fit_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_1_cont.txt", "rb") as fp:   
    x_NMMSO_1 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_1_cont.txt", "rb") as fp:   
    f_NMMSO_1 = pickle.load(fp)  
    
      
#############################################  NMMSO 10 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO2_cont.txt", "rb") as fp:   
    design_dict_NMMSO2 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO2_cont.txt", "rb") as fp:   
    fit_dict_NMMSO2  = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_2_cont.txt", "rb") as fp:   
    x_NMMSO_2 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_2_cont.txt", "rb") as fp:   
    f_NMMSO_2 = pickle.load(fp)  


#############################################  NMMSO 11 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO3_cont.txt", "rb") as fp:   
    design_dict_NMMSO3 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO3_cont.txt", "rb") as fp:   
    fit_dict_NMMSO3 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_3_cont.txt", "rb") as fp:   
    x_NMMSO_3 = pickle.load(fp)     
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_3_cont.txt", "rb") as fp:   
    f_NMMSO_3 = pickle.load(fp)      

#############################################  NMMSO 00 without penalty


with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO0_cont.txt", "rb") as fp:   
    design_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO0_cont.txt", "rb") as fp:   
    fit_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_0_cont.txt", "rb") as fp:   
    x_NMMSO_0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_0_cont.txt", "rb") as fp:   
    f_NMMSO_0 = pickle.load(fp)  
######################################################################

x_NMMSO_1[21]
f_NMMSO_1[21]
x_NMMSO_1[18]
f_NMMSO_1[18]

    
######################################################################  NMMSO bar chart    


for i in x_NMMSO_MI:
    i[5] = round(i[5])
    i[6] = round(i[6])



    
gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []
f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []
for j,i in enumerate(x_NMMSO_MI):
    print(i)
    print(j)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(f_NMMSO_MI[j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(f_NMMSO_MI[j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(f_NMMSO_MI[j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(f_NMMSO_MI[j])

x = ['1','2','3','0']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(0, 31, 2.0),fontsize=9)
plt.xticks(fontsize=20)
bar = plt.bar(x, height=y,color= 'royalblue')
bar[0].set_color('purple')
plt.title('NMMSO: MI',fontsize=25)
plt.savefig('Desktop/cont_figs/NMMSO_MI_frequency_cont.eps', format='eps',bbox_inches='tight')


##########################################################   Box plot- NMMSO -GG


st.median(f_NMMSO_0)
st.median(f_NMMSO_1)
st.median(f_NMMSO_2)
st.median(f_NMMSO_3)

data = [f_NMMSO_1, f_NMMSO_0, f_NMMSO_3, f_NMMSO_2]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '0',
                    '3', '2'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4))
plt.title("NMMSO:GxG",fontsize=25)
plt.savefig('Desktop/cont_figs/NMMSO_GG_boxplot.eps', format='eps',bbox_inches='tight')
plt.show()
    
#########################################################################    PCP CMA-NMMSO  GG 01


#######  NMMSO gxg data

fn = f_NMMSO_1
xn = x_NMMSO_1

########  CMA-ES gxg data

xc = x_CMA_1
fc = f_CMA_1




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
plt.savefig('Desktop/cont_figs/NMMSO_GG_PCP_cont.eps', format='eps',bbox_inches='tight')
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
plt.savefig('Desktop/cont_figs/CMAES_GG_PCP_cont.eps', format='eps',bbox_inches='tight')
plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    