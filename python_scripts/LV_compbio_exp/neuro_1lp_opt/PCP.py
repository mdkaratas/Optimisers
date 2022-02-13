#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 22:33:18 2021

@author: melikedila
"""
import matplotlib 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.cm as cm
from matplotlib import ticker
from colour import Color


####  Use only for PCPs




#########  NMMSO mi data
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
        
#######  NMMSO gxg data

gate_1g = x_NMMSO_1
gate_3g= x_NMMSO_3
gate_0g = x_NMMSO_0
gate_2g = x_NMMSO_2

f_g1g = f_NMMSO_1
f_g3g =  f_NMMSO_3
f_g0g = f_NMMSO_0
f_g2g = f_NMMSO_2        
 

########  CMA-ES MI data

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

########  CMA-ES gxg data

gate_1g = x_CMA_1
gate_3g= x_CMA_3
gate_0g = x_CMA_0
gate_2g = x_CMA_2

f_g1g = f_CMA_1
f_g3g =  f_CMA_3
f_g0g = f_CMA_0
f_g2g = f_CMA_2   

##########  NMMSO MI Penalty data

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

########  NMMSO gxg Penalty data

gate_1g = x_NMMSO_1_P
gate_3g= x_NMMSO_3_P
gate_0g = x_NMMSO_0_P
gate_2g = x_NMMSO_2_P

f_g1g = f_NMMSO_1_P
f_g3g =  f_NMMSO_3_P
f_g0g = f_NMMSO_0_P
f_g2g = f_NMMSO_2_P


########  CMA-MI Penalty data

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
        
########  CMA-GxG Penalty data  

gate_1g = x_CMA_1_P
gate_3g= x_CMA_3_P
gate_0g = x_CMA_0_P
gate_2g = x_CMA_2_P

f_g1g = f_CMA_1_P
f_g3g =  f_CMA_3_P
f_g0g = f_CMA_0_P
f_g2g = f_CMA_2_P   
      
        
#######    create color list for all 8 graphs  when MI included as in CMA-ES
f = f_g0.copy()
for i in f_g1:
    f.append(i)

for i in f_g2:
    f.append(i)
for i in f_g3:
    f.append(i)
for i in f_g1g:
    f.append(i)
for i in f_g2g:
    f.append(i)
for i in f_g3g:
    f.append(i)  
for i in f_g0g:
    f.append(i)       

f = sorted(f_g1g)    
f = sorted(f)
norm = colors.Normalize(min(f), max(f))
colours = cm.RdBu(norm(f))  # div
c0=[]
c1=[]
c2=[]
c3=[]
c4 =[]
c5 = []
c6 = []
c7 = []
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
        
for i in range(len(f_g0g)):
    if f_g0g[i]==f[f.index(f_g0g[i])]:
        c4.append(colours[f.index(f_g0g[i])]) 
for i in range(len(f_g1g)):        
    if f_g1g[i]==f[f.index(f_g1g[i])]:
        c5.append(colours[f.index(f_g1g[i])])   
for i in range(len(f_g2g)):        
    if f_g2g[i]==f[f.index(f_g2g[i])]:
        c6.append(colours[f.index(f_g2g[i])])   
for i in range(len(f_g3g)):        
    if f_g3g[i]==f[f.index(f_g3g[i])]:
        c7.append(colours[f.index(f_g3g[i])])   
        


######  plot ALL


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
for i in gate_1g:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g1g} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
#norm = colors.Normalize(min(f_g1), max(f_g1))
#colours = cm.RdBu(norm(f_g1))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c5[idx])
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
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]],fontsize=30)
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
#sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
#position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
#plt.colorbar(sm,cax=position)# pad = 0.15 shrink=0.9 colorbar length
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figures/CMA_P_GG,01,PCP_.eps', format='eps',bbox_inches='tight')
plt.show()    










        
#######   NMMSO 

f = []

for i in f_g1g:
    f.append(i)
for i in f_g2g:
    f.append(i)
for i in f_g3g:
    f.append(i)  
for i in f_g0g:
    f.append(i)       
    
f = sorted(f)
norm = colors.Normalize(min(f), max(f))
colours = cm.RdBu(norm(f))  # div
c0=[]
c1=[]
c2=[]
c3=[]

    
for i in range(len(f_g0g)):
    if f_g0g[i]==f[f.index(f_g0g[i])]:
        c0.append(colours[f.index(f_g0g[i])]) 
for i in range(len(f_g1g)):        
    if f_g1g[i]==f[f.index(f_g1g[i])]:
        c1.append(colours[f.index(f_g1g[i])])   
for i in range(len(f_g2g)):        
    if f_g2g[i]==f[f.index(f_g2g[i])]:
        c2.append(colours[f.index(f_g2g[i])])   
for i in range(len(f_g3g)):        
    if f_g3g[i]==f[f.index(f_g3g[i])]:
        c3.append(colours[f.index(f_g3g[i])])  

gate_1 = x_NMMSO_1
gate_3 = x_NMMSO_3
gate_0 = x_NMMSO_0
gate_2 = x_NMMSO_2


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
for i in gate_0g:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_g0g} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(min(f_g1), max(f_g1))
colours = cm.RdBu(norm(f_g1))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)
min_max_range = {} # Get min, max and range for each column
for col in cols: # Normalize the data for each column
    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= c0[idx])
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
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]],fontsize=30)
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
#sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
#position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
#plt.colorbar(sm,cax=position)# pad = 0.15 shrink=0.9 colorbar length
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figures/NMMSO:GxG_P,00,PCP_.eps', format='eps',bbox_inches='tight')
plt.show()    

### plot colorbar
fig,ax = plt.subplots()
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=min(f), vmax=max(f)))
plt.colorbar(sm,ax=ax,orientation="horizontal")
ax.remove()
plt.savefig('Desktop/Figures/colorbarextract.eps', format='eps',bbox_inches='tight')    


##########################   Comparable GxG runs

#######  NMMSO gxg data

gate_1 = x_NMMSO_1_P
f_g1 = f_NMMSO_1_P
f = sorted(f_g1)
m = 0.0939922480620155 #min(f_g1)
fn =[]
xn = []
for i in range(len(f_g1)):
    if f_g1[i]==m:
        fn.append(f_g1[i])
        xn.append(gate_1[i])

########  CMA-ES gxg data

gate_1g = x_CMA_1
f_g1g = f_CMA_1
f = sorted(f_g1g)
m = 0.06734496124031007#min(f_g1g)
fc =[]
xc = []
for i in range(len(f_g1g)):
    if f_g1g[i]==m:
        fc.append(f_g1g[i])
        xc.append(gate_1g[i])


###########################################  only g1

f = []
f = sorted(f_g1)
for i in f_g1:
    f.append(i)

norm = colors.Normalize(min(f[0]), max(f[-1]))
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
###########################################  buu
### cloorlist
f = []

for i in fn:
    f.append(i)
for i in fc:
    f.append(i)
    
    
f = sorted(f)
norm = colors.Normalize(min(f), max(f))
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
for i in xn:
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
norm = colors.Normalize(min(fc), max(fc))
colours = cm.RdBu(norm(fn))  # divides the bins by the size of the list normalised data
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
    norm_min = lb[col]#df[cols[dim]].min()
    norm_range = np.ptp([ub[col],lb[col]])#np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks-1)
    ticks = [round(norm_min + norm_step * i, 1) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels,fontsize =20)
for dim, ax in enumerate(axes):
    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
    set_ticks_for_axis(dim, ax, ticks=10) # here ticks 6 has 6 labels on y axis
    ax.set_xticklabels([cols[dim]],fontsize =20)
ax = plt.twinx(axes[-1]) # Move the final axis' ticks to the right-hand side
dim = len(axes)
ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
set_ticks_for_axis(dim, ax, ticks=6)
ax.set_xticklabels([cols[-2], cols[-1]],fontsize=30)
ax.xaxis.set_tick_params(labelsize=50) ##  ax.tick_params(axis = 'both', which = 'major', labelsize = 24) sagdaki y labellarini degistiriyor
plt.subplots_adjust(wspace=0) # Remove space between subplots
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=f_g1[0], vmax=f_g1[-1]))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
plt.colorbar(sm,cax=position)# pad = 0.15 shrink=0.9 colorbar length
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figures/NMM_MI_P_ext.eps', format='eps',bbox_inches='tight')
plt.show()    
        
x =[]
for i in xn:
    print(i[0])
    x.append(i)
        
    
##########################################  

c1=[]
c2=[]

gate_1g = x_CMA_1_P
f = f_CMA_1_P

f = []

for i in f_CMA_1_P:
    f.append(i)
for i in f_NMMSO_1_P:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0, 4)
colours = cm.RdBu(norm(f))  # div
    
for i in range(len(f_CMA_1_P)):        
    if f_CMA_1_P[i]==f[f.index(f_CMA_1_P[i])]:
        c1.append(colours[f.index(f_CMA_1_P[i])])   
for i in range(len(f_NMMSO_1_P)):        
    if f_NMMSO_1_P[i]==f[f.index(f_NMMSO_1_P[i])]:
        c2.append(colours[f.index(f_NMMSO_1_P[i])])   



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
for i in x_NMMSO_1_P:
    t1.append(i[0])
    t2.append(i[1])
    t3.append(i[2])
    T1.append(i[3])
    T2.append(i[4])
    
data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2, 'Fitness': f_NMMSO_1_P} 
df = pd.DataFrame(data)
cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']
x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(min(f), max(f))
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
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=4))
position=fig.add_axes([0.97,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.4)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) #matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=30)
plt.savefig('Desktop/Figures/NMMSO_GG_P_pcp.eps', format='eps',bbox_inches='tight')
plt.show()    