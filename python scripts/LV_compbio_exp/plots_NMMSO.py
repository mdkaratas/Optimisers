#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:39:05 2021

@author: melikedila
"""
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
from matplotlib.cm import ScalarMappable
import statistics as st
#%%
#####    Readings and the plots

###      NMMSO

# without penalty

#############################################  NMMSO MI without penalty

with open("Desktop/Llyonesse/NMMSO/design_dict_NMMSO_MI.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO_MI.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)
with open("Desktop/Llyonesse/NMMSO/x_NMMSO_MI.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)      
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_MI.txt", "rb") as fp:   
    f_NMMSO_MI  = pickle.load(fp)         
    
#############################################  NMMSO 01 without penalty

with open("Desktop/Llyonesse/NMMSO/design_dict_NMMSO1.txt", "rb") as fp:   
    design_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO1.txt", "rb") as fp:   
    fit_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/x_NMMSO_1.txt", "rb") as fp:   
    x_NMMSO_1 = pickle.load(fp)      
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_1.txt", "rb") as fp:   
    f_NMMSO_1 = pickle.load(fp)  
    
      
#############################################  NMMSO 10 without penalty

with open("Desktop/Llyonesse/NMMSO/design_dict_NMMSO2.txt", "rb") as fp:   
    design_dict_NMMSO2 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO2.txt", "rb") as fp:   
    fit_dict_NMMSO2  = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/x_NMMSO_2.txt", "rb") as fp:   
    x_NMMSO_2 = pickle.load(fp)      
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_2.txt", "rb") as fp:   
    f_NMMSO_2 = pickle.load(fp)  


#############################################  NMMSO 11 without penalty

with open("Desktop/Llyonesse/NMMSO/design_dict_NMMSO3.txt", "rb") as fp:   
    design_dict_NMMSO3 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO3.txt", "rb") as fp:   
    fit_dict_NMMSO3 = pickle.load(fp)      
with open("Desktop/Llyonesse/NMMSO/x_NMMSO_3.txt", "rb") as fp:   
    x_NMMSO_3 = pickle.load(fp)     
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_3.txt", "rb") as fp:   
    f_NMMSO_3 = pickle.load(fp)      

#############################################  NMMSO 00 without penalty


with open("Desktop/Llyonesse/NMMSO/design_dict_NMMSO0.txt", "rb") as fp:   
    design_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO0.txt", "rb") as fp:   
    fit_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/x_NMMSO_0.txt", "rb") as fp:   
    x_NMMSO_0 = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_0.txt", "rb") as fp:   
    f_NMMSO_0 = pickle.load(fp)  


# with penalty

#############################################  NMMSO MI with Penalty

with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P.txt", "rb") as fp:   
    design_dict_NMMSO_MI_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/fit_dict_NMMSO_MI_P.txt", "rb") as fp:   
    fit_dict_NMMSO_MI_P = pickle.load(fp)       
with open("Desktop/Llyonesse/penalty_results/nmmso-p/x_NMMSO_MI_P.txt", "rb") as fp:   
    x_NMMSO_MI_P = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/f_NMMSO_MI_P.txt", "rb") as fp:   
    f_NMMSO_MI_P = pickle.load(fp)    
    
    
 
#############################################  NMMSO 01 with Penalty

with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO1_P.txt", "rb") as fp:   
    design_dict_NMMSO1_P = pickle.load(fp)  
with open("Desktop/Llyonesse/NMMSO/fit_dict_NMMSO1.txt", "rb") as fp:   
    fit_dict_NMMSO1_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/x_NMMSO1_P.txt", "rb") as fp:   
    x_NMMSO_1_P = pickle.load(fp)      
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO1_P.txt", "rb") as fp:   
    f_NMMSO_1_P = pickle.load(fp)  

#############################################  NMMSO 10 with Penalty

with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO2_P.txt", "rb") as fp:   
    design_dict_NMMSO2_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/fit_dict_NMMSO2_P.txt", "rb") as fp:   
    fit_dict_NMMSO2_P  = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/x_NMMSO2_P.txt", "rb") as fp:   
    x_NMMSO_2_P = pickle.load(fp)      
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO2_P.txt", "rb") as fp:   
    f_NMMSO_2_P = pickle.load(fp)  

   
    
#############################################  NMMSO 11 with Penalty

with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P.txt", "rb") as fp:   
    design_dict_NMMSO3_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/fit_dict_NMMSO3_P.txt", "rb") as fp:   
    fit_dict_NMMSO3_P = pickle.load(fp)      
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/x_NMMSO3_P.txt", "rb") as fp:   
    x_NMMSO_3_P = pickle.load(fp)     
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO3_P.txt", "rb") as fp:   
    f_NMMSO_3_P = pickle.load(fp)        
    
#############################################  NMMSO 00 with Penalty

with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO0_P.txt", "rb") as fp:   
    design_dict_NMMSO0_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/fit_dict_NMMSO0_P.txt", "rb") as fp:   
    fit_dict_NMMSO0_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/x_NMMSO0_P.txt", "rb") as fp:   
    x_NMMSO_0_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO0_P.txt", "rb") as fp:   
    f_NMMSO_0_P = pickle.load(fp)     




    
 #%%   
############################################################################################  Plots. 

   
################################################################   Bar chart- NMMSO- MI    


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
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=20)
plt.bar(x, height=y,color= 'royalblue')
plt.title('NMMSO: MI',fontsize=25)
plt.savefig('Desktop/Figures/NMMSO:MI,frequency.eps', format='eps',bbox_inches='tight')




##########################################################   Box plot- NMMSO-MI   # it has only 01 f_g1
#st.median(f_g0)
#st.median(f_g1)
#st.median(f_g2)
#st.median(f_g3)


data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color
ax.set_xticklabels(['1', '0',
                    '3', '2'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,1.1))
plt.title("NMMSO-ES:MI",fontsize=25)
plt.savefig('Desktop/Figures/NMMSO:MI,boxplot.eps', format='eps',bbox_inches='tight')
plt.show()



###########################################################   PCP NMMSO-MI
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

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1} 
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
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
plt.savefig('Desktop/Figures/NMMSO:MI,1,PCP_.eps', format='eps')
plt.show()
############################################################################### PCP 2
import matplotlib.colors as colors
import matplotlib 
from matplotlib import ticker
#######    create color list for all 8 graphs
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


#########
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
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(min(f_g1), max(f_g1))
colours = cm.RdBu(norm(f_g1))  # divides the bins by the size of the list normalised data
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
    

#############################################################################################  GxG NMMSO

##########################################################   Bar chart- NMMSO- GxG  XX  ( all 01)


gate_1 = x_NMMSO_1
gate_3 = x_NMMSO_3
gate_0 = x_NMMSO_0
gate_2 = x_NMMSO_2

f_g1 = f_NMMSO_1
f_g3=  f_NMMSO_3
f_g0 = f_NMMSO_0
f_g2 = f_NMMSO_2



x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('NMMSO: MI')
plt.savefig('Desktop/Figures/NMMSO:GxG,frequency.eps', format='eps')

##########################################################   Box plot- NMMSO- GxG
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
plt.ylim((0,1.1))
plt.title("NMMSO:GxG",fontsize=25)
plt.savefig('Desktop/Figures/NMMSO:GxG,boxplot.eps', format='eps',bbox_inches='tight')
plt.show()


###########################################################   PCP NMMSO- GxG 

#############   01    
    


gate_1 = x_NMMSO_1
gate_3 = x_NMMSO_3
gate_0 = x_NMMSO_0
gate_2 = x_NMMSO_2
f_g0 = f_NMMSO_0
f_g1 = f_NMMSO_1
f_g2 = f_NMMSO_2
f_g3=  f_NMMSO_3



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

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1} 
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
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
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


for i in gate_3:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g3} 
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
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g3), min(f_g3)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
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
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g0} 
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
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g0), min(f_g0)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
plt.title(' g = 00')
plt.savefig('Desktop/Figures/NMMSO:GxG,00,PCP_.eps', format='eps')
plt.show()



############################################################################################
##########################################################   Bar chart- NMMSO- MI  + Penalty  

gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []

f_g1 = []
f_g3=  []
f_g0 = []
f_g2 = []


for j,i in enumerate(x_NMMSO_MI_P):
    print(i)
    print(j)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(f_NMMSO_MI_P[j])
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(f_NMMSO_MI_P[j])
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(f_NMMSO_MI_P[j])
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(f_NMMSO_MI_P[j])


x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=20)
bar = plt.bar(x, height=y,color= 'royalblue')
bar[0].set_color('purple')
plt.title('NMMSO: MI ',fontsize=19)
plt.savefig('Desktop/Figs_compbio/NMMSO_MI_P_bchart.eps', format='eps',bbox_inches='tight')

##########################################################   Box plot- NMMSO-MI + Penalty 


data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color
ax.set_xticklabels(['1', '0',
                    '3', '2'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,1.1))
plt.title("NMMSO-ES:MI",fontsize=20)
plt.savefig('Desktop/Figs_compbio/NMMSO_MI_P_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()

###########################################################   PCP NMMSO-MI + Penalty 

###  01
comb = list(zip(x_NMMSO_MI_P,f_NMMSO_MI_P))
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
plt.savefig('Desktop/Figures/NMMSO:MI+penalty,1,PCP_.eps', format='eps')
plt.show()

######  10

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
plt.title('g = 10 (P)')
cbar = plt.colorbar(ScalarMappable(norm=plt.Normalize(max(f_g2), min(f_g2)), cmap='RdBu'), ax=thePlot)
#cbar.set_label('# fitness values', rotation=270)
cbar.ax.set_yticklabels(['0.08','0.10','0.12','0.14','0.16','0.18'])
plt.savefig('Desktop/Figures/NMMSO:MI+penalty,2,PCP_.eps', format='eps')
plt.show()




#############################################################################################

##########################################################   Bar chart- NMMSO- GxG  + Penalty 



gate_1 = x_NMMSO_1
gate_3 = x_NMMSO_3
gate_0 = x_NMMSO_0
gate_2 = x_NMMSO_2
f_g0 = f_NMMSO_0
f_g1 = f_NMMSO_1
f_g2 = f_NMMSO_2
f_g3=  f_NMMSO_3


x = ['1','3','0','2']
y = [len(gate_1), len(gate_3),len(gate_0),len(gate_2)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('NMMSO: MI, GxG')
plt.savefig('Desktop/Figures/NMMSO:MI,frequency.eps', format='eps')

##########################################################   Box plot- NMMSO- GxG + Penalty 

np.median(f_NMMSO_3_P)
np.median(f_NMMSO_0_P)

data = [f_NMMSO_1_P, f_NMMSO_0_P, f_NMMSO_3_P, f_NMMSO_2_P]
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
plt.title("NMMSO:GxG ",fontsize=20)
plt.savefig('Desktop/Figs_compbio/NMMSO_GG_P_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()


###########################################################   PCP NMMSO- GxG + Penalty 






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



###############  Pairwise


#############################################################################     Pairwise plots   MI

# reorder the list of solutions according to their fitness

comb = list(zip(x_NMMSO_MI,f_NMMSO_MI))
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

with open("Desktop/Llyonesse/f_pairs.txt", "rb") as fp:   
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

n = 30
f_col = []
c = 30
k = 0
for i in range(n):  
    f_col.append(colours[k:k+c])
    k = k+c
    c = c-1



n = 30
fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='lightyellow')   #g, axes = plt.subplots(nrows=30, ncols=30,figsize=(10,10))
gs = fig.add_gridspec(nrows=n, ncols=n)
ax = np.zeros((n, n), dtype=object)
for i in range(n):
    for j in range(0+i,n):
        if j >= i:
            ax[j,i] = fig.add_subplot(gs[j,i])
            ax[j,i].plot(corr_x[i][j-i],linewidth=0.2,color=f_col[i][j-i])
            ax[j,i].axis('off')               
plt.axis('off')
sm = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm)
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
a = plt.colorbar(sm,cax=position,shrink=0.9)# pad = 0.15 shrink=0.9 colorbar length
a.ax.tick_params(labelsize=6) 
plt.savefig('Desktop/Figures/pairwise.eps', format='eps',bbox_inches='tight')
plt.show()









































#######  --quick fix 
f_NMMSO_1s = []
for i in f_NMMSO_1:
    f_NMMSO_1s.append(-i)
    
f_NMMSO_2s = []
for i in f_NMMSO_2:
    f_NMMSO_2s.append(-i)
f_NMMSO_3s = []
for i in f_NMMSO_3:
    f_NMMSO_3s.append(-i)
f_NMMSO_0s = []
for i in f_NMMSO_0:
    f_NMMSO_0s.append(-i)
    
 ##########################################################   Box plot- NMMSO: - Gx G   
data = [f_NMMSO_1s, f_NMMSO_2s, f_NMMSO_3s, f_NMMSO_0s]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("NMMSO:GxG")
plt.savefig('Desktop/Llyonesse/Plots/NMMSO:GxG,boxplot.eps', format='eps')
plt.show()

##########################################################   Bar chart- CMA-ES - MI    
###########   frequency plots - bar chart

gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

for i in x_CMA_MI:
    print(i)
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(neuro1lp_costf(i))
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(neuro1lp_costf(i))
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(neuro1lp_costf(i))
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(neuro1lp_costf(i))


x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,30))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI')
plt.savefig('Desktop/Llyonesse/Plots/CMA-ES:MI,frequency.eps', format='eps')

##########################################################   Box plot- CMA-MI 

data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:MI")
plt.savefig('Desktop/Llyonesse/Plots/CMA:MI,boxplot.eps', format='eps')
plt.show()

##########################################################   Box plot- CMA-Es: - Gx G

data = [f_CMA_1, f_CMA_2, f_CMA_3, f_CMA_0]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:GxG")
plt.savefig('Desktop/Llyonesse/Plots/CMA-ES:GxG,boxplot.eps', format='eps')
plt.show()


##########  PCPs    



















###########   frequency plots - bar chart

gate_1 = []
gate_3 = []
gate_0 = []
gate_2 = []


f_g0 = []
f_g1 = []
f_g2 = []
f_g3 = []

for i in x_CMA_MI_P:
    if i[5:7] == [0, 1]:
        gate_1.append(i)
        f_g1.append(neuro1lp_costf(i))
    if i[5:7] == [1, 1] :
        gate_3.append(i)
        f_g3.append(neuro1lp_costf(i))
    if i[5:7] == [0, 0] :
        gate_0.append(i)
        f_g0.append(neuro1lp_costf(i))
    if i[5:7] == [1, 0]:
        gate_2.append(i)
        f_g2.append(neuro1lp_costf(i))


x = ['1','2','3','0']
y = [len(gate_1), len(gate_2),len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,30))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI, penalty func')
plt.savefig('Desktop/Llyonesse/Plots/CMA-ES:MI + penalty,frequency.eps', format='eps')

########             boxplot
data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 6))

ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data,patch_artist=True, )
ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.ylim((0,4))
plt.title("CMA-ES:MI + penalty func.")
plt.savefig('Desktop/Llyonesse/Plots/CMA:MI,boxplot, penalty func.eps', format='eps')
plt.show()

###############################  NMMSO













     







#%%



#Hepsini acip da combinelama zimbirtisi    



with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_1.txt", "rb") as fp:   
    f_CMA_MI_P_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_2.txt", "rb") as fp:   
    f_CMA_MI_P_2 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_3.txt", "rb") as fp:   
    f_CMA_MI_P_3 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_4.txt", "rb") as fp:   
    f_CMA_MI_P_4 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_5.txt", "rb") as fp:   
    f_CMA_MI_P_5 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P_6.txt", "rb") as fp:   
    f_CMA_MI_P_6 = pickle.load(fp) 
f_CMA_MI_P = f_CMA_MI_P_1 + f_CMA_MI_P_2 + f_CMA_MI_P_3 +f_CMA_MI_P_4+f_CMA_MI_P_5 +f_CMA_MI_P_6

with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P.txt", "wb") as fp:   
    pickle.dump(f_CMA_MI_P, fp)     
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P.txt", "rb") as fp:   
    f_CMA_MI_P = pickle.load(fp)  
    

#### NMMSO pozitife dondurme    
f_CMA_MI_P = []
for i in f_NMMSO_0:
    f_CMA_MI_P.append(-1*i)    
with open("Desktop/Llyonesse/NMMSO/f_NMMSO_0.txt", "wb") as fp:   
    pickle.dump(f_CMA_MI_P, fp)      
    
    
#####   fNMMSO- xNMMSO fix
fn_NMMSO_MI =[]
for i in x_NMMSO_MI:
   i[5] = round(i[5])
   i[6] = round(i[6])
   fn_NMMSO_MI.append(neuro1lp_costf(i))     
   
   
####




with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_1.txt", "rb") as fp:   
    f_CMA_MI_P_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_2_1.txt", "rb") as fp:   
    f_CMA_MI_P_2 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_2_2.txt", "rb") as fp:   
    f_CMA_MI_P_3 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_2_3.txt", "rb") as fp:   
    f_CMA_MI_P_4 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_2_4.txt", "rb") as fp:   
    f_CMA_MI_P_6 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_2_5.txt", "rb") as fp:   
    f_CMA_MI_P_7 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_3.txt", "rb") as fp:   
    f_CMA_MI_P_8 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_4.txt", "rb") as fp:   
    f_CMA_MI_P_9 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_5.txt", "rb") as fp:   
    f_CMA_MI_P_10 = pickle.load(fp)                 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P_6.txt", "rb") as fp:   
    f_CMA_MI_P_11 = pickle.load(fp)


vals = []
for i in f_CMA_MI_P_2.values():
    vals.append(i)
for i in f_CMA_MI_P_3.values():
    vals.append(i)
for i in f_CMA_MI_P_4.values():
    vals.append(i)
for i in f_CMA_MI_P_6.values():
    vals.append(i)    
for i in f_CMA_MI_P_7.values():
    vals.append(i)
for i in f_CMA_MI_P_8.values():
    vals.append(i)
for i in f_CMA_MI_P_9.values():
    vals.append(i)
for i in f_CMA_MI_P_10.values():
    vals.append(i)
for i in f_CMA_MI_P_11.values():
    vals.append(i)          
for i in range(25):
    f_CMA_MI_P_1[i+5] = vals[i]
    
    


x_NMMSO_MI_P = f_CMA_MI_P_1 + f_CMA_MI_P_2 + f_CMA_MI_P_3 + f_CMA_MI_P_4+ f_CMA_MI_P_5+ f_CMA_MI_P_6+f_CMA_MI_P_7+f_CMA_MI_P_8+f_CMA_MI_P_9+f_CMA_MI_P_10+f_CMA_MI_P_11
f_CMA_MI_P = []
for i in f_NMMSO_MI_P:
    f_CMA_MI_P.append(-1*i)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/design_dict_NMMSO_MI_P.txt", "wb") as fp:   
    pickle.dump(f_CMA_MI_P_1, fp)    
    
    
#####

fn_NMMSO_MI =[]
for i in x_NMMSO_MI_P:
   i[5] = round(i[5])
   i[6] = round(i[6])
   fn_NMMSO_MI.append(neuro1lp_costf(i))   
   
f_NMMSO_MI_P = fn_NMMSO_MI

with open("Desktop/Llyonesse/penalty_results/nmmso-p/x_NMMSO_MI_P.txt", "wb") as fp:   
    pickle.dump(x_NMMSO_MI_P, fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/f_NMMSO_MI_P.txt", "wb") as fp:   
    pickle.dump(f_NMMSO_MI_P, fp)     



with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_1.txt", "rb") as fp:   
    f_CMA_MI_P_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_2.txt", "rb") as fp:   
    f_CMA_MI_P_2 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_3.txt", "rb") as fp:   
    f_CMA_MI_P_3 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_4.txt", "rb") as fp:   
    f_CMA_MI_P_4 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_5.txt", "rb") as fp:   
    f_CMA_MI_P_6 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P_6.txt", "rb") as fp:   
    f_CMA_MI_P_7 = pickle.load(fp) 



vals = []
for i in f_CMA_MI_P_2.values():
    vals.append(i)
for i in f_CMA_MI_P_3.values():
    vals.append(i)
for i in f_CMA_MI_P_4.values():
    vals.append(i)
for i in f_CMA_MI_P_6.values():
    vals.append(i)    
for i in f_CMA_MI_P_7.values():
    vals.append(i)  
    
for i in range(25):
    f_CMA_MI_P_1[i+5] = vals[i]    
    
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/design_dict_NMMSO3_P.txt", "wb") as fp:   
    pickle.dump(f_CMA_MI_P_1, fp)       
        
    
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_1.txt", "rb") as fp:   
    f_CMA_MI_P_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_2.txt", "rb") as fp:   
    f_CMA_MI_P_2 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_3.txt", "rb") as fp:   
    f_CMA_MI_P_3 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_4.txt", "rb") as fp:   
    f_CMA_MI_P_4 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_5.txt", "rb") as fp:   
    f_CMA_MI_P_6 = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO_2_P_6.txt", "rb") as fp:   
    f_CMA_MI_P_7 = pickle.load(fp) 




    
f_CMA_MI_P = []    
for i in x_NMMSO_MI_P:
    f_CMA_MI_P.append(-1*i)  

x_NMMSO_MI_P = f_CMA_MI_P_1 + f_CMA_MI_P_2 + f_CMA_MI_P_3 + f_CMA_MI_P_4+ f_CMA_MI_P_6+f_CMA_MI_P_7
f_CMA_MI_P = []
for i in x_NMMSO_MI_P:
    f_CMA_MI_P.append(-1*i)  
with open("Desktop/Llyonesse/penalty_results/nmmso-p/gates_penalty/f_NMMSO2_P.txt", "wb") as fp:   
    pickle.dump(f_CMA_MI_P, fp)       
    
    
#############################################################################     Pairwise plots   MI

# reorder the list of solutions according to their fitness

comb = list(zip(x_NMMSO_MI,f_NMMSO_MI))
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

with open("Desktop/Llyonesse/f_pairs.txt", "rb") as fp:   
    f_pairs = pickle.load(fp)  

    
       
#### plot pairwise  ###  bu bir dursun


plt.figure(figsize=(3,3),dpi=5000)
axes1 = plt.gca()
l =[0]
for i in range(30):
    l.append(l[-1]+31)
    
for i in range(len(sampled_pairs)):
    if i not in l:
        plt.subplot(30,30,i+1)
        plt.plot(list(range(10)),f_pair[i], color='red',linewidth=0.2)
        plt.axis('off')
        plt.grid(True)
plt.savefig('Desktop/Figures/pairwise.eps', format='eps',bbox_inches='tight')

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

n = 30
f_col = []
c = 30
k = 0
for i in range(n):  
    f_col.append(colours[k:k+c])
    k = k+c
    c = c-1



n = 30
fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='lightyellow')   #g, axes = plt.subplots(nrows=30, ncols=30,figsize=(10,10))
gs = fig.add_gridspec(nrows=n, ncols=n)
ax = np.zeros((n, n), dtype=object)
for i in range(n):
    for j in range(0+i,n):
        if j >= i:
            ax[j,i] = fig.add_subplot(gs[j,i])
            ax[j,i].plot(corr_x[i][j-i],linewidth=0.2,color=f_col[i][j-i])
            ax[j,i].axis('off')               
plt.axis('off')
sm = plt.cm.ScalarMappable(cmap='nipy_spectral', norm=norm)
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
a = plt.colorbar(sm,cax=position,shrink=0.9)# pad = 0.15 shrink=0.9 colorbar length
a.ax.tick_params(labelsize=6) 
plt.savefig('Desktop/Figures/pairwise.eps', format='eps',bbox_inches='tight')
plt.show()




##########  bunu Tinkle projesi icin olabilir...
ndim = 4
nsamp = 1000
np.random.seed(10)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps2 = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)

#Get the getdist MCSamples objects for the samples, specifying same parameter
#names and labels; if not specified weights are assumed to all be unity
names = ["x%s"%i for i in range(ndim)]
labels =  ["x_%s"%i for i in range(ndim)]
samples = MCSamples(samples=samps,names = names, labels = labels)
samples2 = MCSamples(samples=samps2,names = names, labels = labels)

#Triangle plot
g = plots.getSubplotPlotter()
g.triangle_plot([samples, samples2], filled=True)

