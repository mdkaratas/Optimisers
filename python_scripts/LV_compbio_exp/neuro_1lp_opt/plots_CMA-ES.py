#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:52:44 2021

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
#####    Readings and the plots

#%%
#####   CMA-ES

# without penalty

#############################################  CMA-ES MI without penalty

with open("Desktop/Llyonesse/continuous_fcn_results/neuro1lp/x_CMA_MI_cont.txt", "rb") as fp:   
    x_CMA_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/continuous_fcn_results/neuro1lp/f_CMA_MI_cont.txt", "rb") as fp:   
    f_CMA_MI = pickle.load(fp)  
    
#############################################  CMA-ES 01 without penalty

with open("Desktop/Llyonesse/continuous_fcn_results/cont/x_CMA_1.txt", "rb") as fp:   
    x_CMA_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/continuous_fcn_results/cont/f_CMA_1.txt", "rb") as fp:   
    f_CMA_1 = pickle.load(fp) 
    
#############################################  CMA-ES 10 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_CMA_2.txt", "rb") as fp:   
    x_CMA_2 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_CMA_2.txt", "rb") as fp:   
    f_CMA_2 = pickle.load(fp)  
    
#############################################  CMA-ES 11 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_CMA_3.txt", "rb") as fp:   
    x_CMA_3 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_CMA_3.txt", "rb") as fp:   
    f_CMA_3 = pickle.load(fp)  
    
#############################################  CMA-ES 00 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_CMA_0.txt", "rb") as fp:   
    x_CMA_0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_CMA_0.txt", "rb") as fp:   
    f_CMA_0 = pickle.load(fp)  


    
    
    
#####################################################################################################



##########################################################   Bar chart- CMA-ES - MI    


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []

for j,i in enumerate(x_CMA_MI):
    print(i)
    print(j)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(f_CMA_MI[j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(f_CMA_MI[j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(f_CMA_MI[j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(f_CMA_MI[j])


x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(0, 31, 2.0),fontsize=9)
plt.xticks(fontsize=20)
bar = plt.bar(x, height=y,color= 'royalblue')
bar[0].set_color('purple')
plt.title('CMA-ES: MI',fontsize=25)
plt.savefig('Desktop/cont_figs/CMAES_MI_frequency.eps', format='eps',bbox_inches='tight')

##########################################################   Box plot- CMA-ES - MI


data = [f_g1, f_g3, f_g0, f_g2]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '0', '2'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:MI",fontsize=25)
plt.savefig('Desktop/cont_figs/CMAES_MI_boxplot.eps', format='eps',bbox_inches='tight')
plt.show()
###########################################################   PCP CMA-ES - MI    
comb = list(zip(x_CMA_MI,f_CMA_MI))
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

######   01


T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_1:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 01')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g1), min(f_g1)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:MI,1,PCP_.eps', format='eps')
plt.show()




### 10



T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_2:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g2} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 10')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g2), min(f_g2)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:MI,2,PCP_.eps', format='eps')
plt.show()



### 11



T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_3:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g3} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,24))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 11')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g3), min(f_g3)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:MI,3,PCP_.eps', format='eps')
plt.show()


### 00



T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_0:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g0} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,24))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 00')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g0), min(f_g0)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:MI,0,PCP_.eps', format='eps')
plt.show()


#############################################################################################  CMA-ES

##########################################################   Bar chart- CMA-ES- GxG  XX  ( all 01)


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []




x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('NMMSO: MI')
plt.savefig('Desktop/Figures/NMMSO:MI,frequency.eps', format='eps')

##########################################################   Box plot- CMA-ES- GxG
st.median(f_CMA_1)
st.median(f_CMA_0)
st.median(f_CMA_2)
st.median(f_CMA_3)

data = [f_CMA_1, f_CMA_3, f_CMA_0, f_CMA_2]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '0', '2'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:GxG",fontsize=25)
plt.savefig('Desktop/cont_figs/CMA_GG_boxplot_neuro1lp.eps', format='eps',bbox_inches='tight')
plt.show()
###########################################################   PCP CMA-ES- GxG 

#############   01    
    


gate_1 = x_CMA_1
gate_3 = x_CMA_3
gate_0 = x_CMA_0
gate_2 = x_CMA_2
f_g0 = f_CMA_0
f_g1 = f_CMA_1
f_g2 = f_CMA_2
f_g3=  f_CMA_3



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
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,12))
plt.ylabel('Parameter values') 
plt.grid(False)
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g1), min(f_g1)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.06' ,'0.08','0.10','0.12','0.14','0.16','0.18'])
plt.title(' g = 01')
plt.savefig('Desktop/Figures/NMMSO:GxG,01,PCP_.eps', format='eps')
plt.show()

###### 10

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]
for i in gate_2:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g2} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,12))
plt.ylabel('Parameter values') 
plt.grid(False)
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g2), min(f_g2)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
#cbar.ax.set_yticklabels(['0.06' ,'0.08','0.10','0.12','0.14','0.16','0.18'])
plt.title(' g = 10')
plt.savefig('Desktop/Figures/NMMSO:GxG,10,PCP_.eps', format='eps')
plt.show()

###### 11

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]
for i in gate_3:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g3} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g3), min(f_g3)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
#cbar.ax.set_yticklabels(['0.06' ,'0.08','0.10','0.12','0.14','0.16','0.18'])
plt.title(' g = 11')
plt.savefig('Desktop/Figures/NMMSO:GxG,11,PCP_.eps', format='eps')
plt.show()


#####  00


T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]
for i in gate_0:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g0} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g0), min(f_g0)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
#cbar.ax.set_yticklabels(['0.06' ,'0.08','0.10','0.12','0.14','0.16','0.18'])
plt.title(' g = 00')
plt.savefig('Desktop/Figures/NMMSO:GxG,00,PCP_.eps', format='eps')
plt.show()



############################################################################################
##########################################################   Bar chart- CMA-ES- MI  + Penalty  


gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []

for j,i in enumerate(x_CMA_MI_P):
    print(i)
    print(j)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(f_CMA_MI_P[j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(f_CMA_MI_P[j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(f_CMA_MI_P[j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(f_CMA_MI_P[j])

x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(0, 31, 2.0),fontsize=15)
plt.xticks(fontsize=20)
bar = plt.bar(x, height=y,color= 'royalblue')
bar[0].set_color('purple')
plt.title('CMA-ES: MI ',fontsize=19)
plt.savefig('Desktop/Figs_compbio/CMA_MI_P_bchart.eps', format='eps',bbox_inches='tight')

##########################################################   Box plot- CMA-MI + Penalty 


data = [f_g1, f_g3,f_g2,  f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '2', '0'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:MI",fontsize= 20)
plt.savefig('Desktop/Figs_compbio/CMA_MI_P_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()

###########################################################   PCP CMA-MI + Penalty 
comb = list(zip(x_CMA_MI_P,f_CMA_MI_P))
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
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,12))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 01 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g1), min(f_g1)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
plt.savefig('Desktop/Figures/CMA:MI+penalty,1,PCP_.eps', format='eps')
plt.show()






#############################################################################################

##########################################################   Bar chart- CMA- GxG  + Penalty xxx


gate_1 = x_CMA_1_P
gate_3 = x_CMA_3_P
gate_0 = x_CMA_0_P
gate_2 = x_CMA_2_P

f_g1 = f_CMA_1_P
f_g3=  f_CMA_3_P
f_g0 = f_CMA_0_P
f_g2 = f_CMA_2_P




x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA:GxG (P)')
plt.savefig('Desktop/Figures/CMA:GxG +Penalty,frequency.eps', format='eps')

##########################################################   Box plot- CMA-ES- GxG + Penalty 



data = [f_g1,f_g3,f_g2,f_g0  ]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '2', '0'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:GxG",fontsize=20)
plt.savefig('Desktop/Figs_compbio/CMA_GG_P_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()
###########################################################   PCP CMA- GxG + Penalty 

##  10


T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_2:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g2} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 10 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g2), min(f_g2)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:GxG,2+Penalty,PCP_.eps', format='eps')
plt.show()

##   01

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_1:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,20))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 01 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g1), min(f_g1)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:GxG,1+Penalty,PCP_.eps', format='eps')
plt.show()

##   11

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_3:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g3} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,24))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 11 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g3), min(f_g3)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:GxG,3+Penalty,PCP_.eps', format='eps')
plt.show()



##   00

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_0:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g0} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,24))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 00 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g0), min(f_g0)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:GxG,0+Penalty,PCP_.eps', format='eps')
plt.show()































### 11



T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_3:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': f_g3} 
df = pd.DataFrame(data)
cmap = plt.get_cmap('RdBu')  #hot,Spectral,coolwarm,YlOrRd,jet  Reds arttikca deger de artiyor.
#cmap = plt.cm.jet(np.linspace(0,1,len(f_NMMSO_MI)))
# Make the plot
thePlot= parallel_coordinates(df, 'Fitness', colormap=cmap) #colormap=plt.get_cmap("hot")
thePlot.get_legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.ylim((0,24))
plt.ylabel('Parameter values') 
plt.grid(False)
plt.title('g = 11')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g3), min(f_g3)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.09','0.10','0.11','0.12','0.13','0.14','0.15'])
plt.savefig('Desktop/Figures/CMA:MI,3,PCP_.eps', format='eps')
plt.show()





# Calling DataFrame constructor  
#df = pd.DataFrame()  



name = []
for i in x_NMMSO_MI:
    if i[5:7] == [0, 1]:
        name.append(1)
    if i[5:7] == [1, 1] :
        name.append(3)
    if i[5:7] == [0, 0] :
        name.append(0)
    if i[5:7] == [1, 0]:
        name.append(2)


data = { 'Design': x_NMMSO_MI, 'Fitness': f_NMMSO_MI, 'Gates': name}  
df = pd.DataFrame(data)
print(df)

sns.boxplot( x=df["Gates"], y=df["Fitness"] );
plt.savefig('Desktop/Figures/NMMSO:MI,boxplot_.eps', format='eps')
plt.show()








###################   pairwise plot for CMA- GG

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



comb = list(zip(x_CMA_1_P,f_CMA_1_P))
sorted_list = sorted(comb, key=lambda x: x[1])
xlist = []
flist =[]
for i in sorted_list:
    xlist.append(i[0])
    flist.append(i[1])

# obtain combinations of two of all the solutions
comb_two = []
for i in range(len(xlist)):
    for j in range(i,len(xlist)):  #simply start from i+1 not to include solution itself
        comb_two.append([xlist[i],xlist[j]])

# linearly spaced 10 points between all combinations of two of solutions in each dimension (5) (since last 2 are all the same in this case)
n_dim =5       
sampled_pairs = []  
for i in range(len(comb_two)):
    l = []
    for j in range(n_dim):
        l.append(list(np.linspace(comb_two[i][0][j], comb_two[i][1][j], 10)))
    sampled_pairs.append(l)
    
# for fitness-- go NMMSO GxG

def fitness(inputparams,gates=[0, 1]):
  inputparams = list(inputparams)
  inputparams = matlab.double(inputparams)
  gates = matlab.double(gates)
  cost = eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
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
        for n in range(5):
            inputparams.append(float(k[n][t]))
        tot_input.append(inputparams)
        l_loc.append(fitness(inputparams,gates=[0, 1]))    
    f_pairs.append(l_loc)
    print("--- %s seconds ---" % (time.time() - start_time))   

with open("Desktop/Llyonesse//Neuro_1lp_res/f_pairs_CMA.txt", "wb") as fp:   
    pickle.dump(f_pairs, fp)  
with open("Desktop/Llyonesse//Neuro_1lp_res/f_pairs_CMA.txt", "rb") as fp:   
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

norm = colors.Normalize(min(f_range), max(f_range))
colours = cm.nipy_spectral(norm(f_range))  # div  RdYlBu gnuplot gist_ncar  nipy_spectral hsv

norm2 = colors.Normalize(min(flist), max(flist))
colours2 = cm.nipy_spectral(norm(flist))  # div  RdYlBu gnuplot gist_ncar  nipy_spectral hsv

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
            ax[i,j].set_ylim([0.06,0.67])
            #ax[j,i].set_title('gs[0, :]')
            ax[i,j].plot(corr_x[i][j-i],linewidth=0.75,color=f_col[i][j-i])
            ax[i,j].axis('off')       
        if j == i:
             ax[j,i] = fig.add_subplot(gs[j,i])  
             ax[j,i].plot((0.5), (0.5), 'o', color=colours2[i])
             ax[j,i].axis('off')

                 
             
plt.axis('off')
sm = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm)
position=fig.add_axes([0.99,0.3,0.03,0.45]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
a = plt.colorbar(sm,cax=position,pad = 0.8,shrink=0.4)# pad = 0.15 shrink=0.9 colorbar length
#sm2 = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm2)
#position2 =fig.add_axes([1.1,0.05,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
#a2 = plt.colorbar(sm2,cax=position2,shrink=0.9)# pad = 0.15 shrink=0.9 colorbar length
#a2.ax.tick_params(labelsize=6) 
plt.savefig('Desktop/Figs_neuro2lp/pairwise_.eps', format='eps',bbox_inches='tight')
plt.show()


###############################################################################   extract 2 comparable sols

c1=[]
c2=[]

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[6]

#gate_1g = globals()['x_CMAES_%s' % savename]
gate_1 =[] 
gate_1.append(x_CMA_1_P[26]) 
gate_1.append(x_CMA_1_P[28])


fn = []
fn.append(f[6])
fn.append(f[11])



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





    
    