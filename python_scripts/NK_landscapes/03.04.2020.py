# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:34:01 2020

@author: mk633
"""

import numpy as np
import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
#from time import time

np.random.seed(0)  ### 2 randomnesss (int matrix + U(0,1) bit contributions)
random.seed(0)


def hammingDistance(a, b) : 
  
    x = a ^ b  
    ones = 0
  
    while (x > 0) : 
        ones += x & 1
        x >>= 1
      
    return ones 

### 1) Creating Interaction Matrix


def imatrix_rand(K, N):
    #indx = list(range(N))      
    Int_matrix_rand = np.zeros((N, K+1), dtype= int)
    for i in range(N):
        ls = list(range(N))
        ls.remove(i)
        Int_matrix_rand[i] = [i]+random.sample(ls, K)
    return(Int_matrix_rand)
    
    
### 2)  Generating Landscape Matrix
    
N = 3
K = 2
lnd_num = 30 # number of landscapes to be genrated

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

        
print(NK_land)
print(Landscape_data)
print(Landscape_data[0, :, -1], np.unique(Landscape_data[0, :, -1], return_counts=False).shape) # checking if the fitness values are repeated


### 3) Find the basin of attraction


# Compute the Look-up matrix
lt = np.array([[i^2**j for j in range(N)] for i in range(2**N) ])

# define the mean fitness value index inside the Landscape_data
fv_idx = N*2

# define the basin_mat storing the basins
basin_mat = {}


# ( repeat the following steps to search for M local optima
#M = 20 )


#all starting pts
for i in range(2**N):
    # ( init_idx = np.random.choice(2**N)  )
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
            
# ( create the edges           
#edg_mat = np.array([[basin_mat[key][-1], basin_mat[in_key][-1], 0] for key in basin_mat.keys() \
#           for in_key in basin_mat.keys() if key!=in_key])   )

print('basin_mat: \n', basin_mat)


### 4)  Merging basins


basin_mat_mer = {}
c = 0
for k, v in basin_mat.items():
    if(basin_mat[k]=='deleted'):
        continue
    temp = np.array([basin_mat[k_in][-1] for k_in,v_in in basin_mat.items() if(k_in!=k and basin_mat[k_in]!='deleted')]) # values of lo
    idx = np.array([k_in for k_in,v_in in basin_mat.items() if(k_in!=k and basin_mat[k_in]!='deleted')]) ## indicies of lo (keys)
    comp = temp==basin_mat[k][-1]
    if(np.any(comp)):
        ls = []
        for idx_k in idx[comp]:
            ls = ls + basin_mat[idx_k][:-1]  # takes the all except for last
            basin_mat[idx_k]= 'deleted'      #merged dict will be deleted
            
            
        basin_mat_mer[str(c)] = np.unique(ls + basin_mat[k][:-1]).tolist()+basin_mat[k][-1:]
        c+=1
    else:
        basin_mat_mer[str(c)] = basin_mat[k]
        c+=1


            
basin_mat = basin_mat_mer
print(basin_mat)


### 5) Nodes


l_opt = [[v[-1],Landscape_data[0, v[-1], -1]] for k,v in basin_mat.items() ]
print(l_opt)


### 6) Basin transitions

basin_edges = []
for key, val in basin_mat.items():
    print(key, val)
    for in_key, in_val in basin_mat.items():
        if (in_key==key or val[-1]==in_val[-1]):
            continue
            
        # compute the p(bi-->bj)
        bi_bj = [np.sum([1/N for s_p in in_val if any(lt[s_p, :]==s)]) for s in val ]
        if(np.mean(np.array(bi_bj))!=0):
            basin_edges.append([val[-1], in_val[-1], np.mean(np.array(bi_bj))])
        
        
        #bi_bj = []
        #for s in val:
        #    s_bj = np.sum([1/N for s_p in in_val if any(lt[s_p, :]==s)])
        #    bi_bj.append(s_bj)
        #    print(s_bj)
basin_edges



### 7) Escape Edges


#escape_edges = []
#D = 1

escape_edges = {D: [] for D in range(1, 3)}

for D in range(1, 3):
    for key, val in basin_mat.items():
        print(key, val)
        for in_key, in_val in basin_mat.items():
            if (in_key==key or val[-1]==in_val[-1]):
                continue  

            # compute the p(bi-->bj)
            wij = np.sum([1 for s_p in in_val if(hammingDistance(s_p, val[-1])<=D)])
            if(wij):
                escape_edges[D].append([val[-1], in_val[-1], wij])        
#escape_edges[D].append([(1,1,1.0),(2,2,1.0),(4,4,1.0),(7,7,1.0)])               
 
# check if the fitness values in the l_opt are the same for more than a node 
# replace the identical values with dissimlar labels
for cont in l_opt:
    basin_edges.append([cont[0],cont[0],1.0])                
                
labels = []
c=1
for cont in l_opt:
    if(str(round(cont[1], 2)) in labels):
        labels.append(str(round(cont[1], 2))+'_'+str(c))
        c+=1
    else:
        labels.append(str(round(cont[1], 2)))
        
print('lables are: ', labels)
           
mapping = {cont[0]: labels[i] for i, cont in enumerate(l_opt)}
print('mapping is: ', mapping)


    
### 8) Constructing Network      

#G = nx.DiGraph()
G = nx.MultiDiGraph()

# Constructing network with basin edges
#str(round((l_opt)[:, 1],2)))
#rl_opt = np.array(l_opt)[:, 1]
#r_opt= np.round(rl_opt,2)
G.add_nodes_from(np.array(l_opt)[:, 0])
#G.add_nodes_from(np.array(r_opt))

G.add_weighted_edges_from(basin_edges)
#G.add_weighted_edges_from([(0.89,0.89,1.0), (0.59,0.59,1.0)])

plt.subplot(131)
node_color = ['green' for v in G]
#edge_color = ['blue' for e in G]-- #G.edges[1, 2]['color'] = "red"
#node_color[np.argmax(np.array(l_opt)[:, 1])]= 'red'
node_color[-1]= 'red'
#node_color[np.argmax(labels)]= 'red'

nx.relabel_nodes(G, mapping, copy=False)
nx.draw_networkx(G, with_label = True, width= np.array(basin_edges)[:, 2]*N , node_size = np.array(l_opt)[:, 1]*900 , node_color =node_color, connectionstyle='arc3, rad = 0.3')   # nx.draw_networkx(G, node_size = node_size, node_color = node_color, alpha = 0.7, with_labels = True, width = edge_width, edge_color ='.4', cmap = plt.cm.Blues)        

print('# of edges: {}'.format(G.number_of_edges()))
print('# of nodes: {}'.format(G.number_of_nodes()))

plt.title('Basin transition edges', size=10)






# Constructing network with escape edges

for D in range(1, 3):
    G_esc = nx.MultiDiGraph()
    G_esc.add_nodes_from(np.array(l_opt)[:, 0])
    G_esc.add_weighted_edges_from(escape_edges[D])
   # G_esc.add_weighted_edges_from([(1,1,1.0),(2,2,1.0),(4,4,1.0),(7,7,1.0)])
    plt.subplot(1, 3, D+1)

    
    
    nx.relabel_nodes(G_esc, mapping, copy=False)
    nx.draw_networkx(G_esc, with_label = True, width= np.array(escape_edges[D])[:, 2] , node_size = np.array(l_opt)[:, 1]*900, node_color = node_color, connectionstyle='arc3, rad = 0.3') 
    plt.title('Escape edges (D= %01d)' %(D), size=10)
    
plt.savefig('LONs.png', format='PNG')

#print('The neighbors of the first node is: \n', list(G.neighbors('0.58')))  # to find the neighbours
#h = G.degree['0.58'] # degree of node
print('The nodes in the network are: \n',list(G.nodes))
#print('The edges in the network are: \n',list(G.edges))


print('Local optima with their mean fitness values: \n', l_opt)
                

H = nx.path_graph(10)
#G.add_nodes_from(H)
#G.graph['day'] = "Monday" #you can modify attributes later



#ls=[]
#for key, val in escape_edges.items():
    #for cont in val:
        #print(cont[2])
        #ls.append(cont[2])
        
#print(ls)






















