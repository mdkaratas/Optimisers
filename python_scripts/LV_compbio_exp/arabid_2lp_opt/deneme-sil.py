#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:18:38 2022

@author: mkaratas
"""


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

# ub_y = [24.0,24.0,24.0,24.0,24.0,24.0,12.0,12.0,12.0,1.0,1.0,1.0,1.0,4.0,4.0]
# lb_y = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


cols = ['$τ_1$','$τ_2$','$τ_3$','$τ_4$', '$τ_5$','$τ_6$','$τ_7$','$τ_8$','$τ_9$','$T_1$','$T_2$','$T_3$','$T_4$','$l_1$','$l_2$']



x = [i for i, _ in enumerate(cols)]
norm = colors.Normalize(0,8) #(min(f), max(f))  !! color...
colours = cm.RdBu(norm(f))  # divides the bins by the size of the list normalised data
fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(12,8)) # Create (X-1) sublots along x axis
fig.text(0.03, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)

ub_y = [24.0,24.0,24.0,24.0,24.0,24.0,12.0,12.0,12.0,1.0,1.0,1.0,1.0,4.0,4.0]
lb_y = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

min_max_range = list()
for i in range(len(lb_y)):
    min_max_range.append((lb_y[i], ub_y[i], ub_y[i]-lb_y[i]))



# # Normalize the data sets
# norm_data_sets = list()
# for ds in data_sets:
#     nds = [(value - min_max_range[dimension][0]) / 
#             min_max_range[dimension][2] 
#             for dimension,value in enumerate(ds)]
#     norm_data_sets.append(nds)
# data_sets = norm_data_sets

for col in cols: # Normalize the data for each column
    min_max_range[col] = [lb[col], ub[col], np.ptp([ub[col],lb[col]])]
    df[col] = np.true_divide(df[col] - lb[col], np.ptp([ub[col],lb[col]])) 

# Plot the datasets on all the subplots
# for i, ax in enumerate(axes):
#     for dsi, d in enumerate(data_sets):
#         ax.plot(x, d, style[dsi])
#     ax.set_xlim([x[i], x[i+1]])

for i, ax in enumerate(axes): # Plot each row
    for idx in df.index:
        im = ax.plot(df.loc[idx, cols], color= rgb[1][idx])
    ax.set_xlim([x[i], x[i+1]])


# Set the x axis ticks 
for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
    ticks = len(axx.get_yticklabels())
    labels = list()
    step = (ub_y[dimension]-lb_y[dimension]) / (ticks - 1)
    mn   = lb_y[dimension]
    for i in xrange(ticks):
        v = mn + i*step
        labels.append('%4.2f' % v)
    axx.set_yticklabels(labels)












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
#plt.savefig('Desktop/cont_figs/CMAES_GG_%s_pcp_naxes.eps'%gate, format='eps',bbox_inches='tight')
plt.show()  