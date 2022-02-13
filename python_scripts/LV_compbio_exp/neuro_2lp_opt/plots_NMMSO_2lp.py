#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 23:54:51 2022

@author: melikedila
"""


import matlab.engine
import numpy as np
import pickle
import time
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
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
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
#####   NMMSO


#############################################  NMMSO MI 

with open("Desktop/Llyonesse/Neuro_2lp_res/NMMSO_MI_neuro2lp/design_dict_NMMSO_MI_neuro2lp.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)   
with open("Desktop/Llyonesse/Neuro_2lp_res/NMMSO_MI_neuro2lp/fit_dict_NMMSO_MI_neuro2lp.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)     
with open("Desktop/Llyonesse/Neuro_2lp_res/NMMSO_MI_neuro2lp/f_NMMSO_MI_neuro2lp.txt", "rb") as fp:   
    f_NMMSO_MI = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_2lp_res/NMMSO_MI_neuro2lp/x_NMMSO_MI_neuro2lp.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)    
    
#############################################  NMMSO GG

n = 5
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(32):
    gate = gatesm[k]
    savename = f"{gate}" 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/NMMSO_GG_neuro2lp/design_dict_{savename}_NMMSO.txt", "rb") as fp:   
        globals()['design_dict_NMMSO_%s' % gate] = pickle.load(fp)   
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/NMMSO_GG_neuro2lp/fit_dict_{savename}_NMMSO.txt", "rb") as fp:   
        globals()['fit_dict_NMMSO_%s' % gate] = pickle.load(fp)     
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/NMMSO_GG_neuro2lp/x_NMMSO_{savename}_NMMSO.txt", "rb") as fp:   
        globals()['x_NMMSO_%s' % gate] = pickle.load(fp) 
    with open(f"Desktop/Llyonesse/Neuro_2lp_res/NMMSO_GG_neuro2lp/f_NMMSO_{savename}_NMMSO.txt", "rb") as fp:   
        globals()['f_NMMSO_%s' % gate]= pickle.load(fp) 
#%%
##########################################################   Bar chart- NMMSO - MI    

for inputparams in x_NMMSO_MI:
    inputparams[8] = round(inputparams[8])
    inputparams[9] = round(inputparams[9])
    inputparams[10] = round(inputparams[10])
    inputparams[11] = round(inputparams[11])
    inputparams[12] = round(inputparams[12])


for k in range(32):
    gate = gatesm[k]
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(x_NMMSO_MI):
        if i[8:19] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(f_NMMSO_MI[j])
            
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
    
x_list = ['3','7','15','5', '29',
 '31',
 '13',
 '9',
 '6',
 '17']

y_list = [11,6,5,2,1,1,1,1,1,1]
#x_list = x_list[0:11]
#y_list = y_list[0:11]

fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=12,rotation=90)
bar = plt.bar(x_list, height=y_list,color= 'royalblue')
bar[0].set_color('purple')
plt.title('NMMSO: MI',fontsize=25)
plt.savefig('Desktop/Figs_neuro2lp/NMMSO_MI_frequency.eps', format='eps',bbox_inches='tight')



#%%
##########################################################   Box plot- NMMSO - GG  


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
plt.savefig('Desktop/Figs_neuro2lp/NMMSO_GG_2lp_boxplt.eps', format='eps',bbox_inches='tight')
plt.show()


##############################  pairwise NMMSO g6 00110

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


for i in range(len(globals()['f_NMMSO_%s' % gate])):
    globals()['f_NMMSO_%s' % gate][i] = -1 * globals()['f_NMMSO_%s' % gate][i]


gate_1g = globals()['x_NMMSO_%s'% savename]
fit = globals()['f_NMMSO_%s' % savename]



comb = list(zip(globals()['x_NMMSO_%s' % savename],fit))
sorted_list = sorted(comb, key=lambda x: x[1])
xlist = []
flist_nmm =[]
for i in sorted_list:
    xlist.append(i[0])
    flist_nmm.append(i[1])

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

with open("Desktop/f_pairs_nmm_2lp_6.txt", "wb") as fp:   
    pickle.dump(f_pairs, fp)  
with open("Desktop/f_pairs_nmm_2lp_6.txt", "rb") as fp:   
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
#norm = colors.Normalize(min(f_range), max(f_range))
#f_r = [0.5* i for i in f_range]
colours = cm.nipy_spectral(norm(f_range))  # div  RdYlBu gnuplot gist_ncar  nipy_spectral hsv

f = []
for i in flist_nmm:
    f.append(i)
for j in flist_cma:    
    f.append(j)
    
f = sorted(f)    
norm2 = colors.Normalize(min(f), max(f))
colours2 = cm.nipy_spectral(norm2(f))  # div  RdYlBu gnuplot gist_ncar  nipy_spectral hsv

c1=[]
c2=[]
    
for i in range(len(flist_nmm)):        
    if flist_nmm[i]==f[f.index(flist_nmm[i])]:
        c1.append(colours2[f.index(flist_nmm[i])])   
for i in range(len(flist_cma)):        
    if flist_cma[i]==f[f.index(flist_cma[i])]:
        c2.append(colours2[f.index(flist_cma[i])])  




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
loc1 = np.array([0.30000000000000004,1.3,2.1,2.9,3.6,4.4,5.2,6,6.8,7.6,8.5,9.3,10.1,10.9,11.7,12.5,13.3,14.1,14.9,15.6,16.3,17,17.8,18.5,19.1,20,20.8,21.5,22.2,23])
labelsa = labels
#plt.xticks([x-0.7 for x in list(range(1,len(labels)+1))], labels, rotation='horizontal',fontsize=4)
plt.xticks(loc1, labels, rotation='horizontal',fontsize=4)
#plt.yticks([x-0.7 for x in list(range(1,len(labels)+1))], labelsy, rotation='horizontal',fontsize=4)
plt.yticks(loc1, labelsy, rotation='horizontal',fontsize=4)
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
             ax[j,i].plot((0.5), (0.5), 'o', color=c1[i])
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
plt.savefig('Desktop/pairwise_2lp_nmmsg6_.eps', format='eps',bbox_inches='tight')
plt.show()


######################################################################################################  pairwise end







