#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:36:21 2022

@author: melikedila
"""

with open("Desktop/for_MI/design_dict_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)  
    



with open("Desktop/for_MI/f_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   
    f_CMAES_MI = pickle.load(fp)  
    
    
    
    
with open("Desktop/for_MI/f_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    f_NMMSO_MI = pickle.load(fp)      


with open("Desktop/for_MI/fit_dict_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)      

   

with open("Desktop/for_MI/x_CMAES_MI_cts_arabid2lp.txt", "rb") as fp:   
    x_CMAES_MI = pickle.load(fp)      

opt = 'CMAES'

with open("Desktop/for_MI/x_NMMSO_MI_cts_arabid2lp.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)      

n = 8
n_gates = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  

for k in range(2**8):
    gate = gatesm[k]
    globals()['gate_%s' % k] = []
    globals()['f_g%s' % k] = []

    
    for j,i in enumerate(x_CMAES_MI):
        i[15:23] = [int(i[a]) for a in range(15,23)]
        if i[15:23] == gate:
            globals()['gate_%s' % k].append(i)
            globals()['f_g%s' % k].append(f_CMAES_MI[j])
            
x = []  
y = []   
for k in range(2**8): 
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
for i in range(2**n_gates):
    x_list.append(x[2**n_gates-1-i])
    y_list.append(y[2**n_gates-1-i])
    
x_list = x_list[0:60]
y_list = y_list[0:60]

fig, axes = plt.subplots(figsize=(7,5), dpi=100)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('gates',fontsize=20)
plt.ylim((0,31))
plt.yticks(np.arange(min(y), 31, 2.0),fontsize=15)
plt.xticks(fontsize=6,rotation=90)
bar = plt.bar(x_list, height=y_list,color= 'royalblue')
#bar[36].set_color('purple')
#bar[39].set_color('purple')
plt.title('%s: MI'%opt,fontsize=25)
