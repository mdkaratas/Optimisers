# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:37:18 2021

@author: mk633
"""
######################################################  Basin size Ackley

################################################# general functions

import numpy as np
import time
import itertools
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
from statistics import mean,stdev 
           
def optima_merge(keys, basin, threshold):
    merged_basin = {}
    merged_bool = [False]* len(basin.keys())
    distances = euclidean_distances(keys, keys)
    condition = distances<=threshold
    for i, row in enumerate(condition):                                
        indices = [j for j, elem in enumerate(row) if (elem and j>i)]    
        if(len(indices)!=0):
            merged_bool[i] = True
            merge = False
            for idx in indices:
                #basin[i] = np.vstack((basin[i], basin[idx])) 
                if (not merged_bool[idx]):
                    merge = True
                    basin[i] = np.vstack((basin[i], basin[idx]))
                    merged_bool[idx] = True
              
            if(merge):
                merged_basin[i] = basin[i]
        else:
            if(not merged_bool[i]):  
                merged_basin[i] = basin[i]

    return merged_basin

def update_nodes(xvals_list, fvals_post , fvals_list, nodes_list, threshold):
    edge_indices = []
    xvals = np.array(xvals_list)
    nodes = np.array(nodes_list)
    dist = euclidean_distances(xvals, nodes)
    dist_bool = dist<=threshold
    for i, row in enumerate(dist_bool):
        idx = np.where(row)[0]
        if(idx.size==0):
           nodes_list.append(xvals[i])
           fvals_list.append(fvals_post[i])
           edge_indices.append(len(nodes_list)-1)
        else:
           edge_indices.append(idx[0])
           
    return edge_indices
def ackley(x):
    return -20* np.exp(-0.2 * np.sqrt( (1/x.shape[0]) * np.sum(x**2) ) ) - \
            np.exp( (1/x.shape[0]) * np.sum( np.cos(2*np.pi*x)) ) + 20 + np.e
            
######################################### nm icin siralama

           
def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)


bh_b=[]
nm_b=[]
##################################################  NM
c = 0
while c<100:
    nonzdelt =0.05
    zdelt = 0.00025
    ndim = 2
    k = 1
    st = {}
    threshold = 10e-5
    fvals = []
    xvals = []
    
    permutations = np.array(list(itertools.product(range(2), repeat=2)))
    permutations[permutations==0] = -1 
    step = np.eye(ndim) * nonzdelt
    rep = 100 
    init_pts = []
    for j in range(rep):
        x0_cam_0 = np.random.uniform(low=-3, high=3, size=(k, 1))
        x0_cam_1 = np.random.uniform(low=-2, high=2, size=(k, 1))
        init_simplex = np.zeros((ndim+1, ndim))
        random_pick = np.arange(2**ndim)
        np.random.shuffle(random_pick)
        i = random_pick[0]
        init_simplex[0] = np.array([x0_cam_0, x0_cam_1]).ravel()
        init_simplex[init_simplex==0] = zdelt
        init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[3].reshape(-1,1)* step
        lb = (-3,-2)
        ub = (3,2)
        n= 2
        for t in range(n):
            for u in range(n+1):
                while (init_simplex[u,t] < lb[t] or init_simplex[u,t] > ub[t]):
                    xt_cam_t = np.random.uniform(low=lb[t], high=ub[t], size=(k, 1)) 
                    init_simplex[0,t] = xt_cam_t
                    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[3].reshape(-1,1)* step
        init_pts.append(init_simplex[0])            
        OptimizeResult = minimize(camel_back, init_simplex[0], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': True})
        #iterx.append(OptimizeResult.allvecs)
        xvals.append(OptimizeResult.x) 
        fvals.append(OptimizeResult.fun)
        st[j] = init_simplex[0]
    
    st = optima_merge(xvals, st, threshold)     
    #print('st: ', st)
    
    # create the initial list of nodes
    nodes_list = [xvals[k] for k,v in st.items()]
    fvals_list = [fvals[k] for k,v in st.items()]
    fvals_list = [round(num, 12) for num in fvals_list]
    st_list= []
    for i,k in st.items():
        st_list.append(len(k))
    
    
    
    ## nm icin siralama
    
    
    opt = np.column_stack((fvals_list, nodes_list,st_list))
    opt = opt[np.argsort(opt[:, 1])]
    fvals_list = opt[:,0]
    nodes_list = opt[:,1:3]
    st_list = opt[:,3]
    fvals_list = list(fvals_list)
    nodes_list = list(nodes_list)
    st_nm = list(st_list)
    
    
    nm_b.append(st_nm)
    
    
    ####################################  BH
    
    k=1
    ndim=2
    threshold = 10e-5
    st = {}
    fvals = []
    xvals = []
    for j in range(len(init_pts)):
        x0 = init_pts[j]
        res = minimize(camel_back, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
        xvals.append(res.x) 
        fvals.append(res.fun)
        st[j] = x0
        
        
    st = optima_merge(xvals, st, threshold)
    
    
    nodes_list = [xvals[k] for k,v in st.items()]
    fvals_list = [fvals[k] for k,v in st.items()]
    fvals_list = [round(num, 12) for num in fvals_list]
    
    st_list= []
    for i,k in st.items():
        st_list.append(len(k))
    opt = np.column_stack((fvals_list, nodes_list,st_list))
    opt = opt[np.argsort(opt[:, 1])]
    fvals_list = opt[:,0]
    nodes_list = opt[:,1:3]
    st_list = opt[:,3]
    fvals_list = list(fvals_list)
    nodes_list = list(nodes_list)
    st_bh = list(st_list)
    
    bh_b.append(st_bh)
    c = c+1



mean_nm = [float(sum(col))/len(col) for col in zip(*nm_b)]
mean_bh = [float(sum(col))/len(col) for col in zip(*bh_b)]

sd_nm = [float(stdev(col)*0.3) for col in zip(*nm_b)]
sd_bh = [float(stdev(col)*0.3) for col in zip(*bh_b)]

fig = plt.figure(figsize=(8,8)) ###plt.figure icinde yazmazsan kare cikmaz
#ax = fig.gca()
ax = fig.add_subplot(111)
#ax.set_xticks(np.arange(0, 1, 0.1))
#ax.set_yticks(np.arange(0, 1., 0.1))
plt.scatter(mean_nm[2], mean_bh[2],c='r', s=50)
plt.scatter(mean_nm[3], mean_bh[3],c='r', s=50)
plt.scatter(mean_nm[0], mean_bh[0],c='b', s=50)
plt.scatter(mean_nm[5], mean_bh[5],c='b', s=50)
plt.scatter(mean_nm[1], mean_bh[1],c='y', s=50)
plt.scatter(mean_nm[4], mean_bh[4],c='y', s=50)
#plt.title('Mean basin size comparison of Nelder-Mead and Basin Hopping',fontsize=15)
plt.xlabel("Mean basin size of Nelder-Mead ",fontsize=20)
plt.ylabel("Mean basin size of Basin-Hopping",fontsize=20)
plt.grid(True)
ax.grid(which='minor', alpha=0.6)
ax.grid(which='major', alpha=0.6)
#plt.errorbar(mean_nm, mean_bh, xerr=sd_nm, yerr=sd_bh, fmt='o',ecolor='gray',color='g')
plt.errorbar(mean_nm[2], mean_bh[2], xerr=sd_nm[2], yerr=sd_bh[2], fmt='o',ecolor='gray',color='r',ms=12)
plt.errorbar(mean_nm[3], mean_bh[3], xerr=sd_nm[3], yerr=sd_bh[3], fmt='o',ecolor='gray',color='r',ms=12)
plt.errorbar(mean_nm[0], mean_bh[0], xerr=sd_nm[0], yerr=sd_bh[0], fmt='o',ecolor='gray',color='b',ms=12)
plt.errorbar(mean_nm[5], mean_bh[5], xerr=sd_nm[5], yerr=sd_bh[5], fmt='o',ecolor='gray',color='b',ms=12)
plt.errorbar(mean_nm[1], mean_bh[1], xerr=sd_nm[1], yerr=sd_bh[1], fmt='o',ecolor='gray',color='y',ms=12)
plt.errorbar(mean_nm[4], mean_bh[4], xerr=sd_nm[4], yerr=sd_bh[4], fmt='o',ecolor='gray',color='y',ms=12)
plt.plot([0, 30], [0, 30], color = 'red', linewidth = 2)
plt.show()