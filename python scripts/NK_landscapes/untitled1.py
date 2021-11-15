# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:18:35 2020

@author: mk633
"""

import numpy as np
import itertools
import os
from time import time
import matplotlib.pyplot as plt
import random

def hammingDistance(a, b) : 
  
    x = a ^ b  
    ones = 0
  
    while (x > 0) : 
        ones += x & 1
        x >>= 1
      
    return ones 

def imatrix_rand(K, N):
    indx = list(range(N))      #  NxNlik bos matrix
    Int_matrix_rand = np.zeros((N, K+1), dtype= int)
    for i in range(N):
        ls = list(range(N))
        ls.remove(i)
        Int_matrix_rand[i] = [i]+random.sample(ls, K)
    return(Int_matrix_rand)
 
N = 6
K = 5
lnd_num = 1 # numer of landscapes to be genrated

NK_land = np.random.rand(2**(K+1), 1)
Landscape_data = np.zeros((lnd_num, 2**N, N*2+1))

for i_out in range(lnd_num):
    
    imatrix = imatrix_rand(K, N)
    print('imatrix_rand: \n', imatrix_rand(K, N))

    for i_in, c2 in enumerate(itertools.product(range(2), repeat=N)):
        #print(list(c2))
        weights = np.power(2, list(range(K, -1, -1)))
        Landscape_data[i_out, i_in, :N] = list(c2)
        Landscape_data[i_out, i_in, N:2*N] = [NK_land[np.sum(np.array(c2)[imatrix[i]]*weights)] for i in range(N)]
        Landscape_data[i_out, i_in, -1] = np.mean(Landscape_data[i_out, i_in, N:2*N])
        
print(Landscape_data[0, :, -1], np.unique(Landscape_data[0, :, -1], return_counts=False).shape)

# compute the look-up matrix which stores the indices of the neighbhours of each binary row 
lt = np.array([[i^2**j for j in range(N)] for i in range(2**N) ])

# define the mean fitness value index inside the Landscape_data
fv_idx = N*2

# define the basin_mat storing the basins
basin_mat = {}
# repeat the following steps to search for M local optima
#M = 20
for i in range(2**N):
#init_idx = np.random.choice(2**N)
    init_idx = i
    i_idx = 0

    # define the basin matrix
    basin_mat[str(i)] = [init_idx]

    # start main loop of the algorithm
    while True:
        print('init_idx value at the beginning of the current iteration:', init_idx, 
              '\nfitness value at init_idx:', Landscape_data[i_idx, init_idx, fv_idx],)

        # find the current neighbours of the selected row (s) based on the look-up table lt computed above
        curr_neigh = lt[init_idx, :]

        # find the neighbour which has the maximum fitness value
        mfit = np.max(Landscape_data[i_idx, curr_neigh, fv_idx])

        # find the index of this neighbour with maximum fitness value
        max_idx = curr_neigh[np.where(Landscape_data[i_idx, curr_neigh, fv_idx]==mfit)][0]

        # compare the found max fitness value with the randomly selected row at the beginning
        if(mfit > Landscape_data[i_idx, init_idx, fv_idx]):
            init_idx = max_idx
            basin_mat[str(i)].append(init_idx) 

            print('curr_neigh: ', curr_neigh, '\nmax fitness value found in the neighbourhood: ', mfit,
                  '\nindex of this maximum value in the table: ', max_idx, '\nnew initial index: ', init_idx,
                 '\n###########################################################\n\n\n')
        else:
            print(' >>> reached local optimum!!')
            break
            
# create the edges           
#edg_mat = np.array([[basin_mat[key][-1], basin_mat[in_key][-1], 0] for key in basin_mat.keys() \
#           for in_key in basin_mat.keys() if key!=in_key])

print('basin_mat: \n', basin_mat)


basin_mat_mer = {}
c = 0
for k, v in basin_mat.items():
    if(basin_mat[k]=='deleted'):
        continue
    temp = np.array([basin_mat[k_in][-1] for k_in,v_in in basin_mat.items() if(k_in!=k and basin_mat[k_in]!='deleted')])
    idx = np.array([k_in for k_in,v_in in basin_mat.items() if(k_in!=k and basin_mat[k_in]!='deleted')])
    comp = temp==basin_mat[k][-1]
    if(np.any(comp)):
        ls = []
        for idx_k in idx[comp]:
            ls = ls + basin_mat[idx_k][:-1]
            basin_mat[idx_k]= 'deleted'
            
            
        basin_mat_mer[str(c)] = np.unique(ls + basin_mat[k][:-1]).tolist()+basin_mat[k][-1:]
        c+=1
    else:
        basin_mat_mer[str(c)] = basin_mat[k]
        c+=1
            
#basin_mat = basin_mat_mer
print(basin_mat)


basin_edges = []
for key, val in basin_mat.items():
    print(key, val)
    for in_key, in_val in basin_mat.items():
        if (in_key==key or val[-1]==in_val[-1]):
            continue
            
        # compute the p(bi-->bj)
        bi_bj = [np.sum([1/N for s_p in in_val if any(lt[s_p, :]==s)]) for s in val ]   
        basin_edges.append([val[-1], in_val[-1], np.mean(np.array(bi_bj))])
        
        
        #bi_bj = []
        #for s in val:
        #    s_bj = np.sum([1/N for s_p in in_val if any(lt[s_p, :]==s)])
        #    bi_bj.append(s_bj)
        #    print(s_bj)
        
basin_edges


escape_edges = []
D = 1
for key, val in basin_mat.items():
    print(key, val)
    for in_key, in_val in basin_mat.items():
        if (in_key==key or val[-1]==in_val[-1]):
            continue
            
        # compute the p(bi-->bj)
        wij = np.sum([1 for s_p in in_val if(hammingDistance(s_p, val[-1])<=D)])  
        escape_edges.append([val[-1], in_val[-1], wij])
        
        
        
escape_edges        