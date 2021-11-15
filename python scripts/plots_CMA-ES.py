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
y = [len(gate_1),len(gate_2) ,len(gate_3),len(gate_0)]
fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI')
plt.savefig('Desktop/Figures/CMA-ES:MI,frequency.eps', format='eps')

##########################################################   Box plot- CMA-ES - MI    

data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)

ax = fig.add_axes([0.1,0.1,0.75,0.75])
bp = ax.boxplot(data,patch_artist=True)  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:MI")
plt.savefig('Desktop/Figures/CMA-ES:MI,boxplot.eps', format='eps')
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

data = [f_CMA_1, f_CMA_3, f_CMA_0, f_CMA_2]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,patch_artist=True)  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '0', '2'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:GxG")
plt.savefig('Desktop/Figures/CMA:GxG,boxplot.eps', format='eps')


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
plt.ylabel('frequency')
plt.xlabel('gates')
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0))
plt.bar(x, height=y,color= '#340B8C')
plt.title('CMA-ES: MI (with Penalty)')
plt.savefig('Desktop/Figures/CMA:MI_P,frequency.eps', format='eps')

##########################################################   Box plot- CMA-MI + Penalty 

data = [f_g1, f_g3, f_g0,f_g2]
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) # ([0, 0, 1, 1]) axis starts at 0.1, 0.1   !!!  bu onemli kaydederken direk axisleri sinirdan baslatmiyor.. ax = fig.add_axes
bp = ax.boxplot(data,patch_artist=True)  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '3',
                    '0', '2'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.ylim((0,4))
plt.title("CMA-ES:MI (P)")
plt.savefig('Desktop/Figures/CMA:MI +penalty,boxplot.eps', format='eps')


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

data = [f_g1, f_g2, f_g3, f_g0]
fig = plt.figure(figsize =(7, 5), dpi=100)

ax = fig.add_axes([0.1,0.1,0.75,0.75])
bp = ax.boxplot(data,patch_artist=True)  # patch_artist=True,  fill with color

ax.set_xticklabels(['1', '2',
                    '3', '0'])

plt.xlabel('LC') 
plt.ylabel('Cost') 
plt.title("CMA-ES:GxG (P)")
plt.savefig('Desktop/Figures/CMA-ES:GxG (P),boxplot.eps', format='eps')


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












    
    