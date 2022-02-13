#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:06:55 2021

@author: melikedila
"""


import math

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
from matplotlib import ticker

comb = list(zip(x_NMMSO_MI,f_NMMSO_MI))
sorted_list = sorted(comb, key=lambda x: x[1])
xlist = []
flist =[]
for i in sorted_list:
    xlist.append(i[0])
    flist.append(i[1])


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []
f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []
for j,i in enumerate(xlist):
    print(i)
    print(j)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(flist[j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(flist[j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(flist[j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(flist[j])

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


for i in gate_1:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])


f_g = sorted(list(set(f_g1)))
df['Fitness'] = pd.cut(df['Fitness'], f_g)
    
#######



import matplotlib.pyplot as plt
import numpy as np


cm = plt.get_cmap('RdBu')
c =[cm(1.*i/30) for i in range(30)]
#############


###################  another try
import matplotlib 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.cm as cm
from matplotlib import ticker
from colour import Color

data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1} 
df = pd.DataFrame(data)


cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
#jet = cm = plt.get_cmap('RdBu')
#cNorm  = colors.Normalize(vmin=0, vmax=f_g1[-1])
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colors.Colormap('RdBu'))
#colours =[cm(2*i/30) for i in range(30)]
norm = colors.Normalize(min(f_g1), max(f_g1))
colours = cm.RdBu(norm(f_g1))
#red = Color("red")
#colours = list(red.range_to(Color("green"),30))

# Create (X-1) sublots along x axis
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8))

fig.text(0.08, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 'xx-large')
# Get min, max and range for each column
# Normalize the data for each column
min_max_range = {}
for col in cols:
    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

# Plot each row
for i, ax in enumerate(axes):
    for idx in df.index:
        #im = ax.plot(df.loc[idx][i], f_g1[idx])
        im = ax.plot(df.loc[idx, cols], color= colours[idx])
    ax.set_xlim([x[i], x[i+1]])
    
# Set the tick positions and labels on y axis for each plot
# Tick positions based on normalised data
# Tick labels are based on original data
def set_ticks_for_axis(dim, ax, ticks):
    min_val, max_val, val_range = min_max_range[cols[dim]]
    step = val_range / float(ticks-1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)

for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]],fontsize =20)
    

# Move the final axis' ticks to the right-hand side
ax = plt.twinx(axes[-1])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]],fontsize=20)
ax.xaxis.set_tick_params(labelsize=50)
##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
# Remove space between subplots
plt.subplots_adjust(wspace=0)
plt.grid(False)

plt.gca().spines['top'].set_visible(False)
# Add legend to plot
#plt.gcf()
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
plt.colorbar(sm)
#plt.title("g = 01")
#[plt.Line2D((0,1),(0,0), color=colours[i]) for i,j in enumerate(df['Fitness'])],bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    #df['Fitness'],    
matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20) 
plt.show()


###  neat PCP
df['Fitness'][idx],
, color= scalarMap.to_rgba(f_g1[idx])

for i, ax in enumerate(axes):
    for idx in df.index:
        #mpg_category = df.loc[idx, 'Fitness']
        im = ax.plot(df.loc[idx][i], f_g1[idx],color= scalarMap.to_rgba(f_g1[idx]))


##########  summary

import matplotlib 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.cm as cm
from matplotlib import ticker
from colour import Color

data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(min(f_g1), max(f_g1))
colours = cm.RdBu(norm(f_g1))
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.08, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 'xx-large')
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= colours[idx])
    ax.set_xlim([x[i], x[i+1]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]],fontsize=20)
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
position=fig.add_axes([0.95,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
plt.colorbar(sm,cax=position)# pad = 0.15 shrink=0.9 colorbar length
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#plt.savefig('Desktop/Figures/NMMSO:GxG,01,PCP_.eps', format='eps')
plt.show()


######## colour list try
f_g1= sorted(f_g1)
norm = colors.Normalize(min(f_g1), max(f_g1))
colours = cm.RdBu(norm(f_g1))  # div
position=fig.add_axes([0.95,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
plt.figure(figsize=(9, 1.5))
plt.colorbar(sm)# 

f = f_g0.copy()
for i in f_g1:
    f.append(i)

for i in f_g2:
    f.append(i)
for i in f_g3:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(min(f), max(f))
colours = cm.RdBu(norm(f))  # div
c0=[]
c1=[]
c2=[]
c3=[]
for i in range(len(f_g0)):
    if f_g0[i]==f[f.index(f_g0[i])]:
        c0.append(colours[f.index(f_g0[i])])
for i in range(len(f_g1)):        
    if f_g1[i]==f[f.index(f_g1[i])]:
        c1.append(colours[f.index(f_g1[i])])   
for i in range(len(f_g2)):        
    if f_g2[i]==f[f.index(f_g2[i])]:
        c2.append(colours[f.index(f_g2[i])])   
for i in range(len(f_g3)):        
    if f_g3[i]==f[f.index(f_g3[i])]:
        c3.append(colours[f.index(f_g3[i])])   
        

# plot only colorbar
fig,ax = plt.subplots()
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
plt.colorbar(sm,ax=ax)
ax.remove()
plt.savefig('Desktop/Figures/onlycolorbar.eps', format='eps',bbox_inches='tight')     
        
        
        