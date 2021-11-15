# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 04:20:05 2021

@author: mk633
"""
#%%
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from networkx.drawing.nx_agraph import to_agraph
#from benchmark_create_edges import optima_merge, update_nodes

    #Toplotsim.append(x)

            
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
           edge_indices.append(len(nodes_list)-1) # if this LO was already found and strored inside nodes list
        else:
           edge_indices.append(idx[0])           
    return edge_indices

#%%
############################################################################  ilk Lyi kaydetme iteration
def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)


nonzdelt =0.05
zdelt = 0.00025
ndim = 2
n = 1
k = 1
st = {}
threshold = 10e-5
fvals = []
xvals = []

#permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
#perm[perm==0] = -1
permutations = np.array(list(itertools.product(range(2), repeat=2)))
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 100 
L = []
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
    L.append(init_simplex[0])            
    OptimizeResult = minimize(camel_back, init_simplex[0], method='nelder-mead', 
                              options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': True})
    #iterx.append(OptimizeResult.allvecs)
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    st[j] = init_simplex

st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
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
st_list = list(st_list)


#%%

#%%
########################################################################### artik edge
global_edges = [] 
nonzdelt = 1.5
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)): 
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(camel_back, nodes_list[p], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': True})
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)

    edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique
    local_edges = [(p, idx, count[j]* 1/(2**ndim)) for j,idx in enumerate(uniqe_edge_indices)] #* 1/((2**ndim)*len(nodes_list))
    nm= np.array(local_edges)
    nm = nm[:,2]
    norm = [float(i)/sum(nm) for i in nm]
    for e,w in enumerate(local_edges):
        w = list(w)
        w[2]= norm[e]
        w= tuple(w)
        local_edges[e]=w
        print(local_edges)    
    global_edges = global_edges + local_edges
        
   
print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)

    
print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)

#from colour import Color
#############################
#import ipycytoscape
G = nx.MultiDiGraph()
ff= np.sort(fvals_list)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#0A0AAA' #blue
c2= '#38eeff' #0080ff' # '#4c4cff' #'#5050F0'
n=len(fvals_list)
node_color= ['#FF0000' if(i==min(fvals_list)) else colorFader(c1,c2,i/n) for i in fvals_list]


node_sz = [v.shape[0]*5 for k,v in st.items()]
print(node_color)
posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [fvals_list[i] for i,elem in enumerate(nodes_list)]
#pos = [(ele[0],ele[1]) for ele in ret] 
network_nodes = [(i, {'color':node_color[i],'style':'filled', 'size':node_sz[i] ,'posx': posx[i], 'posy':posy[i], 'posz': posz[i]\
                      }) for i in np.arange(len(nodes_list))]
#pos = [(elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)]
#pos_dict={}
#for t in range(len(nodes_list)):
    #os_dict[t]= pos[t]
G.add_nodes_from(network_nodes)

for y in range(len(global_edges)):
    G.add_edge(global_edges[y][0],global_edges[y][1],weight= 10*global_edges[y][2]) 

#path = "C:/Users/mk633/Desktop/code/0.5.sixhump_camel.graphml"
#nx.draw(G,cmap=plt.get_cmap('Blues'))
nx.write_graphml(G,'sixhump_1.2,1.graphml')   

#%%  SONRAKI RUNLAR

num_nodes.append(len(fvals_list))

density.append(nx.density(G)) # the ratio of actual edges in the network to all possible edges in the network

degrees_in = [d for n, d in G.in_degree()]
avrg_degree_in = sum(degrees_in) / float(nnodes)
in_deg.append(avrg_degree_in)

degrees_out = [d for n, d in G.out_degree()]
avrg_degree_out = sum(degrees_out) / float(nnodes)
out_deg.append(avrg_degree_out)

c = sorted(nx.degree_centrality(G).items(), key=lambda x : x[1], reverse=True)[:5]
centrality.append(c[0][1]) 

assortativity.append(nx.degree_assortativity_coefficient(G)) 
#%% 
path= []
for i in range(len(fvals_list)):
    if fvals_list[i]==min(fvals_list):
        ind = i
ph = nx.shortest_path_length(G, target=ind)
path.append(1)


