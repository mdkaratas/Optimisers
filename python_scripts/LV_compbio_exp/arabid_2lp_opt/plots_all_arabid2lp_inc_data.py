#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:42:06 2022

@author: mkaratas
"""

import matlab.engine
import numpy as np
import cma
import pickle
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem
from pynmmso.listeners import TraceListener
import itertools
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


n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi

#%%  

# Set required paths
root = '/Users/mkaratas/Desktop/GitHub/'

path= root + r"/BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= path= root + r"/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')['dataLD']
dataDD = eng.load('dataLL.mat')['dataLL']
lightForcingLD = eng.load('lightForcingLD.mat')['lightForcingLD']
lightForcingDD = eng.load('lightForcingLL.mat')['lightForcingLL']


#%%
#  To read all data

read_root = root + "Optimisers/Llyonesse/continuous_fcn_results/"
model_list = {"neuro1lp":2,"neuro2lp": 5, "arabid2lp":8}
optimisers = ["CMAES","NMMSO"]
#optimisers = ["NMMSO"]


for model, n_gates in model_list.items():
    for opt in optimisers:
        with open(read_root + model + "/x_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['x_%s_MI_%s' % (opt,model)] = pickle.load(fp)   
        with open(read_root + model + "/f_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['f_%s_MI_%s' % (opt,model)] = pickle.load(fp) 
        # if opt =="NMMSO":
        #     with open(read_root + model + "/fit_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
        #         globals()['fit_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)   
        #     with open(read_root + model + "/design_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
        #         globals()['design_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)           
        gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))      
        for gate in gatesm: #[3:5]
            with open(read_root + model + f"/x_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'x_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp) 
            with open(read_root + model + f"/f_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'f_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp)
            if opt == "NMMSO":
                with open(read_root + model + "/fit_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['fit_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp)   
                with open(read_root + model + "/design_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['design_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp) 
                    
                    
#########################################################################################################  round last 8 of NMMSO
for inputparams in globals()['x_%s_MI_%s' % ("NMMSO","arabid2lp")]:
    for i in range(8):
        inputparams[15+i] = round(inputparams[15+i])

#########################################################################################################  MI frequency plot- barchart
model =  "NMMSO"  #"CMAES" #  "NMMSO"  #   "NMMSO"  #

for k in range(256):
    gate = gatesm[k]
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(globals()['x_%s_MI_%s' % ("%s"%model,"arabid2lp")]):   # (opt,model)   ###   ("CMAES","arabid2lp")]):
        if i[15:23] == gate:     ####  change here for global
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(globals()['x_%s_MI_%s' % ("%s"%model,"arabid2lp")][j])


#%%
####        Open for MI search trajectories


with open(read_root + "arabid2lp/given_x_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   
     x_CMAES_MI = pickle.load(fp)   
     
with open(read_root + 'arabid2lp/given_x_NMMSO_MI_cts_arabid2lp.txt', "rb") as fp:   
     x_NMMSO_MI = pickle.load(fp)        


with open(read_root + 'arabid2lp/trace_f_CMAES_MI_cts_arabid2lp.txt', "rb") as fp:   
     f_CMAES_MI = pickle.load(fp)       
with open(read_root + 'arabid2lp/trace_f_NMMSO_MI_cts_arabid2lp.txt', "rb") as fp:   
     f_NMMSO_MI = pickle.load(fp)      

#%%

            
x  = []  
y = []   
for k in range(256): 
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
for i in range(256):
    x_list.append(x[255-i])
    y_list.append(y[255-i])
    
count = 0
for i in y_list:
    if i!= 0:
        count = count+1
    
x_list = x_list[0:count]
y_list = y_list[0:count]

fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=12,rotation=90)
bar = plt.bar(x_list, height=y_list,color= 'royalblue')
#bar[1].set_color('purple')
plt.title('%s: MI'%model,fontsize=25)
plt.savefig('Desktop/cont_figs/%s_MI_frequency_arabidopsis_2lp.eps'%(model), format='eps',bbox_inches='tight')


#########################################################################################################    GG boxplot
model = "NMMSO"# "CMAES" #"NMMSO"  #"CMAES" # 
#data = []
xlabs = []
for k in range(256):
    xlabs.append(str(k))
    #data.append(globals()['f_g%s' % k])
for k in range(256):
    gate = gatesm[k]
    for i in range(len(globals()[f'f_%s_%s_%s' % (model,f"arabid2lp",gate)])):
        #print(i)
        globals()[f'f_%s_%s_%s' % (model,f"arabid2lp",gate)][i] = -1 * globals()[f'f_%s_%s_%s' % (model,f"arabid2lp",gate)][i]


 
data = []
for k in range(256):
    gate = gatesm[k]
    globals()['f_g%s' % k] = globals()[f'f_%s_%s_%s' % (model,f"arabid2lp",gate)]
    data.append(globals()['f_g%s' % k])
    
f_median = []
for i in data:
    f_median.append(np.median(i))    
    

comb = list(zip(data,f_median))
sorted_list = sorted(comb, key=lambda x: x[1])
#sorted_ls = sorted_list[0:50]
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
plt.xticks(fontsize=1,rotation=90)
plt.xlabel('LC',fontsize=20) 
plt.ylabel('Cost',fontsize=20) 
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_overall.eps'%model, format='eps',bbox_inches='tight')
plt.show()

###############################  256 gate 5 farkli grupta plot edildi


s1 = sorted_list[0:50]
s2 = sorted_list[50:100]
s3 = sorted_list[100:150]
s4 = sorted_list[150:200]
s5 = sorted_list[200:256]

############################   ilk 50si

val = []
label = []
for i in s1:
    label.append(i[0])
    val.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) 

bp = ax.boxplot(xdata[0:50],notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))
ax.set_xticklabels(label) 
plt.yticks(fontsize=15)
plt.xticks(fontsize=8,rotation=90)
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt.eps'%model, format='eps',bbox_inches='tight')
plt.arrow(18, 5, 0, -1, width = 0.2,color = 'red')
plt.arrow(19, 5, 0, -1, width = 0.2,color = 'red')
plt.arrow(22, 5, 0, -1, width = 0.2,color = 'red')
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_s1.eps'%model, format='eps',bbox_inches='tight')
# show plot
plt.show()


############################   50-100 arasi

val = []
label = []
for i in s2:
    label.append(i[0])
    val.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) 

bp = ax.boxplot(xdata[50:100],notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))
ax.set_xticklabels(label) 
plt.yticks(fontsize=15)
plt.xticks(fontsize=8,rotation=90)
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
# plt.arrow(18, 5, 0, -1, width = 0.2,color = 'red')
# plt.arrow(19, 5, 0, -1, width = 0.2,color = 'red')
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_s2.eps'%model, format='eps',bbox_inches='tight')
# show plot
plt.show()

############################   100-150 arasi

val = []
label = []
for i in s3:
    label.append(i[0])
    val.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) 

bp = ax.boxplot(xdata[100:150],notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))
ax.set_xticklabels(label) 
plt.yticks(fontsize=15)
plt.xticks(fontsize=8,rotation=90)
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
#plt.arrow(25, 6, 0, -1, width = 0.2,color = 'red')
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_s3.eps'%model, format='eps',bbox_inches='tight')
# show plot
plt.show()


############################   150-200 arasi

val = []
label = []
for i in s4:
    label.append(i[0])
    val.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) 

bp = ax.boxplot(xdata[150:200],notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))
ax.set_xticklabels(label) 
plt.yticks(fontsize=15)
plt.xticks(fontsize=8,rotation=90)
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
plt.arrow(14, 6, 0, -1, width = 0.2,color = 'red')
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_s4.eps'%model, format='eps',bbox_inches='tight')
# show plot
plt.show()


############################   200-256 arasi

val = []
label = []
for i in s5:
    label.append(i[0])
    val.append(i[1])
    
fig = plt.figure(figsize =(7, 5), dpi=100)
ax = fig.add_axes([0.1,0.1,0.75,0.75]) 

bp = ax.boxplot(xdata[200:256],notch=False,patch_artist=True,boxprops=dict(facecolor="lightblue"))
ax.set_xticklabels(label) 
plt.yticks(fontsize=15)
plt.xticks(fontsize=8,rotation=90)
plt.ylim((0,8.2))
plt.title("%s:GxG"%model,fontsize=20)
plt.savefig('Desktop/cont_figs/%s_GG_2lp_boxplt_s5.eps'%model, format='eps',bbox_inches='tight')
# show plot
plt.show()


##################################################################################################
##################################################################################################   PCP G 152-3-4-5  ( Altta all combination var colours icin)
##################################################################################################   

modelcma = "CMAES"
modelnmm = "NMMSO"

c1=[]
c2=[]

n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[152]
savename = f"{gate}"


gate_1g = globals()[f'x_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]
f = globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]

f = []

for i in globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]:
    f.append(i)
for i in globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)]:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0,8) #(min(f), max(f)) ...   #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    
for i in range(len(globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)])):        
    if globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)][i])]:
        c1.append(colours[f.index(globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)][i])])   
for i in range(len(globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)])):        
    if globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)][i])]:
        c2.append(colours[f.index(globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)][i])])   

#%%  asagiis tamamen farkli renkler de olsun diye, diger skip edebilirsin
######################################################################## Tum gateler icin all colour- istersen skip et 
###############################################################################   152-3-4-5 den sonra geleni de ekleyebilirsin
modelcma = "CMAES"
modelnmm = "NMMSO"


n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

### First gate 152
GATES = []
for i in range(152,156):
    GATES.append(gatesm[i])


# savename = f"{gate}"


# gate_1g = globals()[f'x_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]
# f = globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]

f = []

for j in GATES:
    gate = j
    for i in globals()[f'f_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]:
        f.append(i)
    for i in globals()[f'f_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)]:
        f.append(i)

    
f = sorted(f)
norm = colors.Normalize(0,8) #(min(f), max(f)) ...   #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div


# Below c list goes like :g = 152 cma - g=152 nmmso , g =153 cma g=153 nmmso
rgb = {} 
opti = [modelcma,modelnmm]
count = 0
for i in range(9):
        rgb[i] =[]

for k in GATES:    
    gate = k
    r = gatesm.index(k)   
    for m in opti:  
        count = count+1
        for i in range(len(globals()[f'f_%s_%s_%s' % (m,f"arabid2lp",gate)])):           
            if globals()[f'f_%s_%s_%s' % (m,f"arabid2lp",gate)][i]==f[f.index(globals()[f'f_%s_%s_%s' % (m,f"arabid2lp",gate)][i])]:
                rgb[count].append(colours[f.index(globals()[f'f_%s_%s_%s' % (m,f"arabid2lp",gate)][i])])
        

#%%

gate = gatesm[155]

###  plot  CMAES
ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 24, '$τ_4$':24, '$τ_5$':24,'$τ_6$':24,'$τ_7$':12,'$τ_8$':12,'$τ_9$':12,
      '$T_1$' :1,'$T_2$' : 1,'$T_3$' :1,'$T_4$' :1,'$l_1$':4,'$l_2$':4}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$τ_6$':0, '$τ_7$':0, '$τ_8$':0,'$τ_9$':0,
      '$T_1$' :0,'$T_2$' : 0, '$T_3$' :0,'$T_4$' :0,'$l_1$':0,'$l_2$':0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
z = 'T_4'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
m = '\u03C4_6'
n = '\u03C4_7'
o = '\u03C4_8'
r = '\u03C4_9'
a = 'l_1'
b = 'l_2'
T1 =[]
T2 =[]
T3 =[]
T4 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]
t6 =[]
t7 =[]
t8 =[]
t9 =[]
l1 =[]
l2 = []
for i in globals()['x_%s_%s_%s' % (modelcma,"arabid2lp",gate)]:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    t6.append(i[5])
    t7.append(i[6])
    t8.append(i[7])
    t9.append(i[8])
    T1.append(i[9])
    T2.append(i[10])
    T3.append(i[11])
    T4.append(i[12])
    l1.append(i[13])
    l2.append(i[14])

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(m): t6, r'${}$'.format(n): t7, r'${}$'.format(o): t8, r'${}$'.format(r): t9,
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,r'${}$'.format(z): T4,
        r'${}$'.format(a):l1,r'${}$'.format(b):l2,'Fitness': globals()['f_%s_%s_%s' %(modelcma,"arabid2lp",gate)]} 

df = pd.DataFrame(data)

ub_y = [24.0,24.0,24.0,24.0,24.0,24.0,12.0,12.0,12.0,1.0,1.0,1.0,1.0,4.0,4.0]
lb_y = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$τ_6$','$τ_7$','$τ_8$','$τ_9$','$T_1$','$T_2$','$T_3$','$T_4$','$l_1$','$l_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,8) #(min(f), max(f))  !! color...
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
        im = ax.plot(df.loc[idx, cols], color= rgb[1][idx])
    ax.set_xlim([x[i], x[i+1]])
    plt.yticks(lb_y,ub_y)
    #ax.set_ylim([lb[i],ub[i]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
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
ax.set_ylim([0,4])
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
#set_ticks_for_axis(dim, ax, ticks=7)
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
#plt.savefig('Desktop/cont_figs/CMAES_GG_%s_pcp_naxes.eps'%gate, format='eps',bbox_inches='tight')
plt.show()    




















#################################################################################################################

gate = gatesm[155]

###  plot  CMAES


ub_y = [25,25,25,25,25,25,12,12,12,1.0,1.0,1.0,1.0,4.0,4.0]
lb_y = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]






ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 24, '$τ_4$':24, '$τ_5$':24,'$τ_6$':24,'$τ_7$':12,'$τ_8$':12,'$τ_9$':12,
      '$T_1$' :1,'$T_2$' : 1,'$T_3$' :1,'$T_4$' :1,'$l_1$':4,'$l_2$':4}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$τ_4$':0, '$τ_5$':0,'$τ_6$':0, '$τ_7$':0, '$τ_8$':0,'$τ_9$':0,
      '$T_1$' :0,'$T_2$' : 0, '$T_3$' :0,'$T_4$' :0,'$l_1$':0,'$l_2$':0}


t = 'T_1'
e = 'T_2'
u = 'T_3'
z = 'T_4'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
m = '\u03C4_6'
n = '\u03C4_7'
o = '\u03C4_8'
r = '\u03C4_9'
a = 'l_1'
b = 'l_2'
T1 =[]
T2 =[]
T3 =[]
T4 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]
t6 =[]
t7 =[]
t8 =[]
t9 =[]
l1 =[]
l2 = []
for i in globals()['x_%s_%s_%s' % (modelcma,"arabid2lp",gate)]:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    t6.append(i[5])
    t7.append(i[6])
    t8.append(i[7])
    t9.append(i[8])
    T1.append(i[9])
    T2.append(i[10])
    T3.append(i[11])
    T4.append(i[12])
    l1.append(i[13])
    l2.append(i[14])

    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(m): t6, r'${}$'.format(n): t7, r'${}$'.format(o): t8, r'${}$'.format(r): t9,
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,r'${}$'.format(z): T4,
        r'${}$'.format(a):l1,r'${}$'.format(b):l2,'Fitness': globals()['f_%s_%s_%s' %(modelcma,"arabid2lp",gate)]} 

df = pd.DataFrame(data)


fig, host = plt.subplots(1, sharey=False, figsize=(14,8))
#fig, host = plt.subplots()
fig.text(0.08, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)


ys = []
for col in cols:
    print(col)
    ys.append(np.array(df[col]))

y0min = ys[0].min()
dy = ys[0].max() - y0min
zs = [ys[0]] + [(y - y.min()) / (y.max() - y.min()) * dy + y0min for y in ys[1:]]
ynames = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$τ_6$','$τ_7$','$τ_8$','$τ_9$','$T_1$','$T_2$','$T_3$','$T_4$','$l_1$','$l_2$']



axes = [host] + [host.twinx() for i in range(len(ys) - 1)]
for i, (ax, y) in enumerate(zip(axes, ys)):  ### i = 15   y = 15 tane 30lu cozum
    print(y)
    #ax.set_ylim(y.min(), y.max())
    ax.set_ylim(lb_y[i], ub_y[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (len(ys) - 1)))

host.set_xlim(0, len(ys) - 1)
host.set_xticks(range(len(ys)))
host.set_xticklabels(ynames)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)


for j in range(len(ys[0])):
    host.plot(range(len(ys)), [z[j] for z in zs], c=rgb[1][j])

sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=8))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.94,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) 
plt.show()
























y0min = 0
dy = 24 - y0min
zs = [ys[0]] + [(y - y.min()) / (y.max() - y.min()) * dy + y0min for y in ys[1:]]
ynames = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$τ_6$','$τ_7$','$τ_8$','$τ_9$','$T_1$','$T_2$','$T_3$','$T_4$','$l_1$','$l_2$']

axes = [host] + [host.twinx() for i in range(len(ys) - 1)]
for i, (ax, y) in enumerate(zip(axes, ys)):   ###   i =14 index, y 30 datas for each 14
    #ax.set_ylim(y.min(), y.max())
    #ax.set_ylim(lb_y[i], ub_y[i])
    #ax.yaxis.set_ticks(np.round(np.arange(lb_y[i], ub_y[i], np.round(ub_y[i]/6)), 1))
    ax.set_ylim(y.min(), y.max())
    
    if ax != host:
        #ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (len(ys) - 1)))

host.set_xlim(0, len(ys) - 1)
#host.set_ylim([0, 24])
host.set_xticks(range(len(ys)))
host.set_xticklabels(ynames)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)


for j in range(len(ys[0])):
    host.plot(range(len(ys)), [z[j] for z in zs], c=rgb[1][j])
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=8))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.94,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) 

#fig.tight_layout()
plt.show()

#################################################################################################





fig, host = plt.subplots()

N = 20
y1 = np.random.uniform(10, 50, N)
y2 = np.sin(np.random.uniform(0, np.pi, N))
y3 = np.random.binomial(300, 0.9, N)
y4 = np.random.pareto(10, N)
y5 = np.random.uniform(0, 800, N)
ys = [y1, y2, y3, y4, y5]
y0min = ys[0].min()
dy = ys[0].max() - y0min
zs = [ys[0]] + [(y - y.min()) / (y.max() - y.min()) * dy + y0min for y in ys[1:]]
ynames = ['P1', 'P2', 'P3', 'P4', 'P5']

axes = [host] + [host.twinx() for i in range(len(ys) - 1)]
for i, (ax, y) in enumerate(zip(axes, ys)):
    ax.set_ylim(y.min(), y.max())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        #ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (len(ys) - 1)))

host.set_xlim(0, len(ys) - 1)
host.set_xticks(range(len(ys)))
host.set_xticklabels(ynames)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)

colors = plt.cm.tab20.colors
for j in range(len(ys[0])):
    host.plot(range(len(ys)), [z[j] for z in zs], c=colors[j % len(colors)])

plt.show()
















###       plot   NMMSO



t = 'T_1'
e = 'T_2'
u = 'T_3'
z = 'T_4'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'
k = '\u03C4_4'
l = '\u03C4_5'
m = '\u03C4_6'
n = '\u03C4_7'
o = '\u03C4_8'
r = '\u03C4_9'
a = 'l_1'
b = 'l_2'
T1 =[]
T2 =[]
T3 =[]
T4 =[]
t1 =[]
t2 =[]
t3 =[]
t4 =[]
t5 =[]
t6 =[]
t7 =[]
t8 =[]
t9 =[]
l1 =[]
l2 = []
for i in globals()[f'x_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)]:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    t4.append(i[3])
    t5.append(i[4])
    t6.append(i[5])
    t7.append(i[6])
    t8.append(i[7])
    t9.append(i[8])
    T1.append(i[9])
    T2.append(i[10])
    T3.append(i[11])
    T4.append(i[12])
    l1.append(i[13])
    l2.append(i[14])
    
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, r'${}$'.format(k): t4, r'${}$'.format(l): t5, 
        r'${}$'.format(m): t6, r'${}$'.format(n): t7, r'${}$'.format(o): t8, r'${}$'.format(r): t9,
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, r'${}$'.format(u): T3,r'${}$'.format(z): T4,
        r'${}$'.format(a):l1,r'${}$'.format(b):l2,'Fitness': globals()[f'x_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)]} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$τ_6$','$τ_7$','$τ_8$','$τ_9$','$T_1$','$T_2$','$T_3$','$T_4$','$l_1$','$l_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,8)#(min(f), max(f))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    #min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = (df[col] - lb[col])/ np.ptp([ub[col],lb[col]])
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= rgb[8][idx])   ###  c2[idx]
    ax.set_xlim([x[i], x[i+1]])
    ax.set_ylim([lb[col],ub[col]])
def set_ticks_for_axis(dim, ax, ticks): # Set the tick positions and labels on y axis for each plot
    min_val, max_val, val_range = min_max_range[cols[dim]] # Tick positions based on normalised data
    step = val_range / float(ticks-1) # Tick labels are based on original data
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
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=8))#(vmin=min(f), vmax=max(f)))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
#plt.savefig('Desktop/cont_figs/NMMSO_GG_%s_pcp_naxes.eps'%gate, format='eps',bbox_inches='tight')
plt.show()    















##################################################################################### 
##################################################################################### 
#####################################################################################  COMPARABLE SOLS
##################################################################################### 
##################################################################################### 

####################################################################################   PCP for gate 152 -   both CMAES and NMMSO [0,0,1,1,1] 

c1=[]
c2=[]

n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[152]
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


#########################################################################################################   es plot
#########################################################################################################

dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)


################################################################################################### su bi dursun calistirma
norm_dataLD = []
norm_dataDD = []


for i in range(len(dataLD)-1):
    norm_dataLD.append(NormalizeData(dataLD_arr[i]))
    norm_dataDD.append(NormalizeData(dataDD_arr[i]))

norm_dataLD.append(dataLD_arr[-1])
norm_dataDD.append(dataDD_arr[-1])

norm_dataLD_ = [sublist[49:] for sublist in norm_dataLD]
norm_dataDD_ = [sublist[49:] for sublist in norm_dataDD]

    
lst = []
lst.append(norm_dataLD_[0])
lst.append(norm_dataLD_[1])
lst = np.array(lst)
lst= lst.T
df = pd.DataFrame(lst, index =[norm_dataLD_[2][i] for i in range(len(norm_dataLD_[2]))],
                                              columns =['mRNA','P'])   
sns.set(rc={'figure.figsize':(18,8.27)},font_scale = 3,style="whitegrid")   
p = sns.lineplot(data=df)   
sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)#plt.legend(loc='upper left')
p.set(xlabel='time steps', ylabel='normalised expression levels',xlim=(20, 125),ylim=(0,1))
plt.savefig('Desktop/neuro1lp_timeseries_LD.eps', format='eps',bbox_inches='tight')
    
    
#  same for DD

lst = []
lst.append(norm_dataDD_[0])
lst.append(norm_dataDD_[1])
lst = np.array(lst)
lst= lst.T
df = pd.DataFrame(lst, index =[norm_dataDD_[2][i] for i in range(len(norm_dataDD_[2]))],
                                              columns =['mRNA','P'])   
sns.set(rc={'figure.figsize':(18,8.27)},font_scale = 3,style="whitegrid")   
p = sns.lineplot(data=df)   
sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)#plt.legend(loc='upper left')
p.set(xlabel='time steps', ylabel='normalised expression levels',xlim=(20, 125),ylim=(0,1))
#########################################################################################################
#########################################################################################################
#########################################################################################################

############################################################################################################## HERE PLOT
n_gates = 8
gatesm = list(map(list, itertools.product([0,1], repeat=n_gates)))   


class Mat_Py:
    def __init__(self):
        self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd = ([] for _ in range(5))
    
    # def update(self, cost, sol_ld, sol_dd, dat_ld, dat_dd):
    #     self.cost.append(cost)
    #     self.sol_ld.append({'x':list(sol_ld['x']._data), 'y':np.asarray(sol_ld['y'], dtype=np.longdouble).tolist()})
    #     self.sol_dd.append({'x':list(sol_dd['x']._data), 'y':list(sol_dd['y']._data)})
    #     self.dat_ld.append({'x':list(dat_ld['x']._data), 'y':list(dat_ld['y']._data)})
    #     self.dat_dd.append({'x':list(sol_dd['x']._data), 'y':list(sol_dd['y']._data)})
        
    def update(self, cost, sol_ld, sol_dd, dat_ld, dat_dd):
        self.cost.append(cost)
        self.sol_ld.append({'x':list(sol_ld['x']._data), 'y':np.asarray(sol_ld['y'], dtype=np.longdouble).tolist()})
        self.sol_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(sol_dd['y'], dtype=np.longdouble).tolist()})
        self.dat_ld.append({'x':list(dat_ld['x']._data), 'y':np.asarray(dat_ld['y'], dtype=np.longdouble).tolist()})
        self.dat_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(dat_dd['y'], dtype=np.longdouble).tolist()})
    
    def get_all(self):
        return [self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd]
         

func_output_CMA = Mat_Py()
func_output_NMMSO = Mat_Py()

gate = gatesm[152]
for idx,sol in enumerate(globals()[f'x_%s_%s_%s' % (modelcma,f"arabid2lp",gate)]):  
    gates = list(gate)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_CMA.update(*eng.getBoolCost_cts_arabid2lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))

[cost_CMA, sol_LD_CMA, sol_DD_CMA, datLD_CMA, datDD_CMA] = func_output_CMA.get_all()
# print(cost_values.sol_dd[0])
# list(cost_values.sol_dd[4]['x']._data)
# cost_values.sol_dd[4]['x'][0]._data
# list(cost_values.sol_dd[4]['x'][0]._data)
# print(cost_values.get_all())
####  NMMSO
gates = gatesm[152]
for idx,sol in enumerate(globals()[f'x_%s_%s_%s' % (modelnmm,f"arabid2lp",gate)]): 
    #print(sol)
    gates = list(gates)
    gates = matlab.double([gates])
    inputparams = sol
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    func_output_NMMSO.update(*eng.getBoolCost_cts_arabid2lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))
    
    
[cost_NMMSO, sol_LD_NMMSO, sol_DD_NMMSO, datLD_NMMSO, datDD_NMMSO] = func_output_NMMSO.get_all()    


################################### new noramlisation--check

#%%    Normalising the data func
  
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


dataLD_arr = np.asarray(dataLD)
dataDD_arr = np.asarray(dataDD)



normalize= []  #336

for i in dataLD_arr[0]:
    normalize.append(i)
for i in dataLD_arr[1]:
    normalize.append(i)   
for i in dataDD_arr[0]:
    normalize.append(i)
for i in dataDD_arr[1]:
    normalize.append(i) 
N = NormalizeData(normalize)

no_dataLD_arr = []
no_dataDD_arr = []
no_dataLD_arr.append(N[0:241])
no_dataLD_arr.append(N[241:482])
no_dataDD_arr.append(N[482:703])
no_dataDD_arr.append(N[703:924])
no_dataLD_arr[0] = no_dataLD_arr[0][48:]
no_dataLD_arr[1] = no_dataLD_arr[1][48:]
no_dataDD_arr[0] = no_dataDD_arr[0][48:]
no_dataDD_arr[1] = no_dataDD_arr[1][48:]
no_dataLD_arr.append(dataLD_arr[-1][48:])
no_dataDD_arr.append(dataDD_arr[-1][48:])

time_swap  = []      # this is to find all 6-18 light switches
tot_hours = 120
for i in range(0,tot_hours+2):
    if i % 2 == 1:
        time_swap.append(6*i)

def partition(values, indices):    # this func is for dividing a list based on another list
    idx = 0
    for index in indices:
        sublist = []
        while idx < len(values) and values[idx] < index:
            sublist.append(values[idx])
            idx += 1
        if sublist:
            yield sublist

df = {'time': no_dataLD_arr[2],
  'A':no_dataLD_arr[0],
  'B':no_dataLD_arr[1]}    
dl_sep = list(partition(df['time'],time_swap))   # Dark-light seperation list



a = []
b = []
for i in range(len(dl_sep)):
    if i % 2 == 0:
        a.append(dl_sep[i])
    elif i % 2 == 1:
        b.append(dl_sep[i])
A =[]
B=[]
for i in range(len(a)):
    #print([a[i][0],a[i][-1]])
    A.append([a[i][0],a[i][-1]])

for i in range(len(b)):

    B.append([b[i][0],b[i][-1]])
    
##########################################################################################   plot discretised TS data

#####################################################################################  now plot CMAES timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
plt.plot(sol_LD_CMA[0]['x'], sol_LD_CMA[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD
for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_CMA[i]['x'], sol_LD_CMA[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')
plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')
#plt.savefig('Desktop/LD_ts_CMA_neuro1lp.eps', format='eps')    
plt.show()
#plt.savefig('x.png')

#####################################################################################  now plot NMMSO timeseries LD

matplotlib.rc('figure', figsize=(20, 7))
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][0] ,linewidth=4,label='mRNA',color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0] #this indexfr the state of the first variable
plt.plot(sol_LD_NMMSO[0]['x'], sol_LD_NMMSO[0]['y'][:2][1]   ,linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')#sol_LD

for i in range(1,len(sol_LD_CMA)):
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][0] ,linewidth=4,color='lightsalmon')#sol_LD_CMA[i]['y'][:2][0]      #this indexfr the state of the first variable
    plt.plot(sol_LD_NMMSO[i]['x'], sol_LD_NMMSO[i]['y'][:2][1]   ,linewidth=4,color='rebeccapurple')#sol_LD_CMA[i]['y'][:2][0]       # this indexes the state of the second variable
    #sol_LD_CMA[i]['x']              # this is the time across
    
    
    # l1 = plt.plot(df['time'], df['A'],linewidth=4,label='mRNA',color='lightsalmon')
    # l2 = plt.plot(df['time'], df['B'],linewidth=4,label='$Protein_{bulk}$',color='rebeccapurple')

plt.xlabel('time (h)',fontsize=25)
plt.ylabel('Normalised expression levels',fontsize=25)
plt.grid(False)
#plt.tick_params(axis ='x')
#plt.tick_params(axis ='y')  # ax4.tick_params(axis ='x', rotation = 45)
plt.xticks(np.arange(18, 121, 6),fontsize= 25, alpha=1)
plt.yticks(fontsize= 25, alpha=1)
plt.tick_params(axis=u'both', which=u'both',length=15,color='black',bottom=True,left=True)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.135),fancybox=False, shadow=False, ncol=2,prop={'size': 20})
plt.margins(x=0)

for i in A:
    plt.axvspan(i[0],i[1], facecolor='grey')
for i in B:
    plt.axvspan(i[0],i[1], facecolor='white')   
plt.savefig('Desktop/LD_ts_NMMSO_neuro1lp.eps', format='eps')        
plt.show()
#plt.savefig('x.png')



















