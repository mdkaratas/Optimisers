# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:16:27 2020

@author: mk633
"""
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize
from pyDOE import *
from scipy.stats.distributions import norm
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph


def himmelblau(X):
    x = X[0]
    y = X[1]
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b


            
def optima_merge(keys, basin, threshold):
    merged_basin = {}
    merged_bool = [False]* len(basin.keys())
    distances = euclidean_distances(keys, keys)
    #condition1 =  distances!=0 
    #condition2 = distances<=threshold
    #condition = condition1 & condition2    #### booleans
    condition = distances<=threshold
    for i, row in enumerate(condition):                                # ith row is already checked so continue jth onwards
        #i is the index of each row(whole matrix), j is column of boolean , elem is elem itself in that row (whole matrix)
        # optimally, we only have to check half of the condition matrix (upper or lower parts around the diagnoal part)
        indices = [j for j, elem in enumerate(row) if (elem and j>i)]    
        if(len(indices)!=0):
            merged_bool[i] = True
            merge = False
            for idx in indices:
                #basin[i] = np.vstack((basin[i], basin[idx])) 
                if(not merged_bool[idx]):
                    merge = True
                    basin[i] = np.vstack((basin[i], basin[idx]))
                    merged_bool[idx] = True
              
            if(merge):
                merged_basin[i] = basin[i]
        else:
            # check whether the key at 'i' was already merged above in comparisons with other keys
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
        # if this LO is not in the current nodes list
        if(idx.size==0):
           nodes_list.append(xvals[i])
           fvals_list.append(fvals_post[i])
           edge_indices.append(len(nodes_list)-1)
        # if this LO was already found and strored inside nodes list
        else:
           edge_indices.append(idx[0])
           
    return edge_indices

############################################################################

nonzdelt = 0.05
zdelt = 0.00025
ndim = 2
n = 1
k = 1
st = {}
threshold = 10e-2
fvals = []
xvals = []
permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 3000
for j in range(rep):
    him_0 = np.random.uniform(low=-5, high=5, size=(k, n))
    him_1 = np.random.uniform(low=-5, high=5, size=(k, n))
    init_simplex = np.zeros((ndim+1, ndim))
    random_pick = np.arange(2**ndim)
    np.random.shuffle(random_pick)
    i = random_pick[0]
    init_simplex[0] = np.array([him_0, him_1]).ravel()
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[i].reshape(-1,1)* step
    init_simplex[0] = np.array([him_0, him_1]).ravel()
    OptimizeResult = minimize(himmelblau, init_simplex[0], method='nelder-mead', 
                              options={'initial_simplex': init_simplex, 'xatol': 1e-18, 'disp': False})
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    st[j] = init_simplex
    
st = optima_merge(xvals, st, threshold)     
#print('st: ', st)
print(st) 
# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
#print(nodes_list)

###########################################################################
# scan all initial points selected randomly above and find LOs from each

fvals_post = []
global_edges = []
init_simplex = np.zeros((ndim+1, ndim))
c = 0 
for k,v in st.items():
    xvals_post = []
    # pick the same initial point we previously started from for this key
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = v[0]
    init_simplex[init_simplex==0] = zdelt
    
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
        init_simplex[0] = v[0]
        OptimizeResult = minimize(himmelblau, v[0], method='nelder-mead', 
                                  options={'initial_simplex': init_simplex, 'xatol': 1e-18, 'disp': False})
        
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)

    # update the nodes list and return existing edge indices
    edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] 

    local_edges = [(c, idx, count[j] * 1/2**ndim) for j, idx in enumerate(uniqe_edge_indices)]
    global_edges = global_edges + local_edges
    c+=1


print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)

##########################################################################
# draw the network

G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
#node_sz = [v.shape[0]*100/rep for k,v in st.items()]
print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])

print(global_edges)
###########################################################################
from chart_studio import plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
N = len(G.nodes)




# define nodes 3D coordinates
pos = [(elem[0], elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)] # fvals_list
Xn = []
Yn = []
Zn = []
for item in pos:
    Xn+= [item[0]]  ## !! not addition adds all x values ..,..,..
    Yn+= [item[1]]
    Zn+= [item[2]]


#Xn= np.random.uniform(coor_max, size=(N)) # x-coordinates
#Yn= np.random.uniform(coor_max, size=(N)) # y-coordinates
#Zn= np.random.uniform(coor_max, size=(N))# z-coordinates

# define mapping from node's names to coordinates
node_IdxToCoor = {}
for i, node_idx in enumerate(G.nodes):
    node_IdxToCoor[node_idx] = [Xn[i], Yn[i], Zn[i]]

# define edges coordinates
Xe=[]
Ye=[]
Ze=[]
for e in global_edges:
    Xe+=[node_IdxToCoor[e[0]][0], node_IdxToCoor[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[node_IdxToCoor[e[0]][1], node_IdxToCoor[e[1]][1], None]# y-coordinates of edge ends
    Ze+=[node_IdxToCoor[e[0]][2], node_IdxToCoor[e[1]][2], None]# z-coordinates of edge ends
    
    
#edge_sz = [item[2] for item in global_edges]
# create edges 3D trace
trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0]*100/rep for k,v in st.items()]  ########
# create nodes 3D trace
trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)
                             ),
               text=[str(n) for n in G.nodes],
               hoverinfo='text'
               )

x_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='x_axis'
          )
y_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='y_axis'
          )
z_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='fitness values'
          )

layout = go.Layout(
         title="Iterated Downhill Simplex- Rastrigin Benchmark Function",
         width=1000,
         height=1000,
         showlegend=True,
         scene=dict(
             xaxis=dict(x_axis),
             yaxis=dict(y_axis),
             zaxis=dict(z_axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    )


data=[trace1, trace2]
fig=go.Figure(data=data, layout=layout)


plot(fig, filename='Nelder-mead.html')