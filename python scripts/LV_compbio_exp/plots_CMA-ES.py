#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:52:44 2021

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
import statistics as st
from matplotlib.cm import ScalarMappable

#%%
#####    Readings and the plots

#%%
#####   CMA-ES

# without penalty

#############################################  CMA-ES MI without penalty

with open("Desktop/Llyonesse/cma/x_CMA_MI.txt", "rb") as fp:   
    x_CMA_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/cma/f_CMA_MI.txt", "rb") as fp:   
    f_CMA_MI = pickle.load(fp)  
    
#############################################  CMA-ES 01 without penalty

with open("Desktop/Llyonesse/cma/x_CMA_1.txt", "rb") as fp:   
    x_CMA_1 = pickle.load(fp) 
with open("Desktop/Llyonesse/cma/f_CMA_1.txt", "rb") as fp:   
    f_CMA_1 = pickle.load(fp) 
    
#############################################  CMA-ES 10 without penalty

with open("Desktop/Llyonesse/cma/x_CMA_2.txt", "rb") as fp:   
    x_CMA_2 = pickle.load(fp)  
with open("Desktop/Llyonesse/cma/f_CMA_2.txt", "rb") as fp:   
    f_CMA_2 = pickle.load(fp)  
    
#############################################  CMA-ES 11 without penalty

with open("Desktop/Llyonesse/cma/x_CMA_3.txt", "rb") as fp:   
    x_CMA_3 = pickle.load(fp)  
with open("Desktop/Llyonesse/cma/f_CMA_3.txt", "rb") as fp:   
    f_CMA_3 = pickle.load(fp)  
    
#############################################  CMA-ES 00 without penalty

with open("Desktop/Llyonesse/cma/x_CMA_0.txt", "rb") as fp:   
    x_CMA_0 = pickle.load(fp)  
with open("Desktop/Llyonesse/cma/f_CMA_0.txt", "rb") as fp:   
    f_CMA_0 = pickle.load(fp)  

# with penalty

#############################################  CMA-ES MI with Penalty


    
with open("Desktop/Llyonesse/penalty_results/CMA_P/x_CMA_MI_P.txt", "rb") as fp:   
    x_CMA_MI_P = pickle.load(fp)     
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_MI_P.txt", "rb") as fp:   
    f_CMA_MI_P = pickle.load(fp)  

     
#############################################  CMA-ES 01 with Penalty

with open("Desktop/Llyonesse/penalty_results/CMA_P/x_CMA_1_P.txt", "rb") as fp:   
    x_CMA_1_P = pickle.load(fp) 
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_1_P.txt", "rb") as fp:   
    f_CMA_1_P = pickle.load(fp) 
    
#############################################  CMA-ES 10 with Penalty

with open("Desktop/Llyonesse/penalty_results/CMA_P/x_CMA_2_P.txt", "rb") as fp:   
    x_CMA_2_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_2_P.txt", "rb") as fp:   
    f_CMA_2_P = pickle.load(fp)  
    
#############################################  CMA-ES 11 with Penalty

with open("Desktop/Llyonesse/penalty_results/CMA_P/x_CMA_3_P.txt", "rb") as fp:   
    x_CMA_3_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_3_P.txt", "rb") as fp:   
    f_CMA_3_P = pickle.load(fp)  
    
#############################################  CMA-ES 00 with Penalty

with open("Desktop/Llyonesse/penalty_results/CMA_P/x_CMA_0_P.txt", "rb") as fp:   
    x_CMA_0_P = pickle.load(fp)  
with open("Desktop/Llyonesse/penalty_results/CMA_P/f_CMA_0_P.txt", "rb") as fp:   
    f_CMA_0_P = pickle.load(fp)      
    
    
    
    
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
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=20)
plt.bar(x, height=y,color= 'royalblue')
plt.title('CMA-ES: MI',fontsize=25)
plt.savefig('Desktop/Figures/CMA-ES:MI,frequency.eps', format='eps',bbox_inches='tight')

##########################################################   Box plot- CMA-ES - MI    

data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '2',
                    '3', '0'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:MI",fontsize=25)
plt.savefig('Desktop/Figures/CMA-ES:MI,boxplot.eps', format='eps',bbox_inches='tight')
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

data = [f_CMA_1, f_CMA_2, f_CMA_3, f_CMA_0]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,notch=True,patch_artist=True,boxprops=dict(facecolor="lightblue"))  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '2',
                    '3', '0'])
plt.yticks(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,4.2))
plt.title("CMA-ES:GxG",fontsize=25)
plt.savefig('Desktop/Figures/CMA:GxG,boxplot.eps', format='eps',bbox_inches='tight')
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










    
    