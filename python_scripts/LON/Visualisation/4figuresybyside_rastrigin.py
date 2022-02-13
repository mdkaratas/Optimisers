# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:36:51 2020

@author: mk633
"""


import numpy as np
import time
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
import math
##### ortak ilk node bulmaca kismi

def rastrigin(x):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * np.cos( 2 * math.pi * x ))
            
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

############################################################################

nonzdelt = 0.05
zdelt = 0.00025
ndim = 2
n = 2
k = 1
st = {}
threshold = 10e-5
fvals = []
xvals = []

permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 100
for j in range(rep):
    x0 = np.random.uniform(low=-5.12, high=5.12, size=(k, n))
    init_simplex = np.zeros((ndim+1, ndim))
    random_pick = np.arange(2**ndim)
    np.random.shuffle(random_pick)
    i = random_pick[0]
    init_simplex[0] = x0
    init_simplex[1:] = init_simplex[0] +  permutations[i].reshape(-1,1)* step
    OptimizeResult = minimize(rastrigin, init_simplex[0], method='nelder-mead', 
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


import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots



fig = make_subplots(rows=2, cols=2,specs=[[{"type": "surface"}, {"type": "scatter3d"}],
                                          [{"type": "scatter3d"}, {"type": "scatter3d"}]], horizontal_spacing = 0.05,vertical_spacing=0.05, subplot_titles=("a) 3-D Landscape visualisation", "b) LON constructed with p = 0.05", "c) LON constructed with p = 1", "d) LON constructed with p = 2"))

x_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='x axis'
          )
y_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='y axis'
          )
z_axis=dict(showbackground=True,
          showline=True,
          zeroline=True,
          showgrid=True,
          showticklabels=True,
          title='fitness values'
          )

layout = go.Layout(
         title="Downhill Simplex six hump camel back",
         width=1000,
         height=1000,
         showlegend=True,
         scene=dict(
             xaxis=dict(y_axis),
             yaxis=dict(x_axis),
             zaxis=dict(z_axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    )

###        1) surface cizimi 




x_interval = (-5.12, 5.12)
y_interval = (-5.12, 5.12)
x = np.linspace(x_interval[0], x_interval[1], 1000)
y = np.linspace(y_interval[0], y_interval[1], 1000)
x,y = np.meshgrid(x,y)
z= 10*2 + ( (x**2 - 10 * np.cos( 2 * np.pi * x ))+ (y**2 - 10 * np.cos( 2 * np.pi * y )))       
#fig.append_trace(go.Figure(data=[go.Surface(z=z, x=y, y=x)],layout=layout), row=1, col=1)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=2, z=0)
)
fig.add_trace(go.Surface(x=x, y=y, z=z) , 1, 1)
#fig.update_layout(scene = dict(zaxis = dict(range=[-1,2])))
#fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)],layout=layout)
#plot(fig, filename='Nelder-mead.html')
###        2) 0.05 olan
#            first edges





global_edges = []
#init_simplex = np.zeros((ndim+1, ndim))
#c = 0
#for k,v in st.items():    
nonzdelt = 0.05
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)): 
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(rastrigin, nodes_list[p], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': False})
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)
    edge_indices = []
    dist = euclidean_distances(xvals_post, nodes_list) #edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    dist_bool = dist<=10e-5
    for i, row in enumerate(dist_bool):
        ind = np.where(row)[0]
        if(ind.size==0):
            print(xvals_post[i],fvals_post[i],i)
            nodes_list.append(xvals_post[i])
            fvals_list.append(fvals_post[i])                
            edge_indices.append(len(nodes_list)-1)
        else: 
            if fvals_list[int(ind[0])]<= fvals_list[p]:
                edge_indices.append(ind[0])
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

G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['blue' if(f==min(fvals_list)) else 'purple' for f in fvals_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from([ (tup[0], tup[1],tup[2]) for tup in global_edges])


pos = [(elem[0], elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)] # fvals_list
Xn = []
Yn = []
Zn = []
for item in pos:
    Xn+= [item[0]]  ## !! not addition adds all x values ..,..,..
    Yn+= [item[1]]
    Zn+= [item[2]]

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

trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0]*5 for k,v in st.items()] 
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
data=[trace1, trace2]
fig.add_traces([go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',
               line=dict(color='rgb(125,125,125)', width=2)),go.Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)
                             ),
               text=[str(n) for n in G.nodes])],rows=[1,1],cols=[2,2])     
#fig.add_trace(go.Figure(data=data, layout=layout),row=1,col=2)  


del G


###        3) 1 olan
#            first edges






global_edges = []
#init_simplex = np.zeros((ndim+1, ndim))
#c = 0
#for k,v in st.items():    
nonzdelt = 1
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)): 
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(rastrigin, nodes_list[p], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': False})
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)
    edge_indices = []
    dist = euclidean_distances(xvals_post, nodes_list) #edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    dist_bool = dist<=10e-3
    for i, row in enumerate(dist_bool):
        ind = np.where(row)[0]
        if(ind.size==0):
            print(xvals_post[i],fvals_post[i],i)
            nodes_list.append(xvals_post[i])
            fvals_list.append(fvals_post[i])                
            edge_indices.append(len(nodes_list)-1)
        else: 
            if fvals_list[int(ind[0])]<= fvals_list[p]:
                edge_indices.append(ind[0])
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


G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['blue' if(f==min(fvals_list)) else 'purple' for f in fvals_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from([ (tup[0], tup[1],tup[2]) for tup in global_edges])


pos = [(elem[0], elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)] # fvals_list
Xn = []
Yn = []
Zn = []
for item in pos:
    Xn+= [item[0]]  ## !! not addition adds all x values ..,..,..
    Yn+= [item[1]]
    Zn+= [item[2]]

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

trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0]*5 for k,v in st.items()] 
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
data=[trace1, trace2]
fig.add_traces([go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',
               line=dict(color='rgb(125,125,125)', width=2)),go.Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)
                             ),
               text=[str(n) for n in G.nodes])],rows=[2,2],cols=[1,1]) 
#fig.add_traces([go.Scatter(x=Ye,y=Xe),go.Scatter(x=Yn,y=Xn)],rows=[2,2],cols=[1,1])     
#fig.add_trace(go.Figure(data=data, layout=layout),row=2,col=1)  





del G

###        4) 2 olan
#            first edges






global_edges = []
#init_simplex = np.zeros((ndim+1, ndim))
#c = 0
#for k,v in st.items():    
nonzdelt = 2
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)): 
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    for i in range(2**ndim): 
        init_simplex[1:] = init_simplex[0] + permutations[i].reshape(-1,1)* step
        OptimizeResult = minimize(rastrigin, nodes_list[p], method='nelder-mead', 
                                  options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': False})
        xvals_post.append(OptimizeResult.x) 
        fvals_post.append(OptimizeResult.fun)
    edge_indices = []
    dist = euclidean_distances(xvals_post, nodes_list) #edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
    dist_bool = dist<=10e-3
    for i, row in enumerate(dist_bool):
        ind = np.where(row)[0]
        if(ind.size==0):
            print(xvals_post[i],fvals_post[i],i)
            nodes_list.append(xvals_post[i])
            fvals_list.append(fvals_post[i])                
            edge_indices.append(len(nodes_list)-1)
        else: 
            if fvals_list[int(ind[0])]<= fvals_list[p]:
                edge_indices.append(ind[0])
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique

    local_edges = [(p, idx, count[j]* 1/(2**ndim)) for j,idx in enumerate(uniqe_edge_indices)] #* 1/((2**ndim)*len(nodes_list))
    nm= np.array(local_edges)
    #nm = nm[:,2]
    #norm = [float(i)/sum(nm) for i in nm]
    #for e,w in enumerate(local_edges):
        #w = list(w)
        #w[2]= norm[e]
        #w= tuple(w)
        #local_edges[e]=w
       # print(local_edges)
    global_edges = global_edges + local_edges

#           second network creation




G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['blue' if(f==min(fvals_list)) else 'purple' for f in fvals_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from([ (tup[0], tup[1],tup[2]) for tup in global_edges])


pos = [(elem[0], elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)] # fvals_list
Xn = []
Yn = []
Zn = []
for item in pos:
    Xn+= [item[0]]  ## !! not addition adds all x values ..,..,..
    Yn+= [item[1]]
    Zn+= [item[2]]

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

trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0]*5 for k,v in st.items()] 
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

fig.add_traces([go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',
               line=dict(color='rgb(125,125,125)', width=2)),go.Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)
                             ),
               text=[str(n) for n in G.nodes])],rows=[2,2],cols=[2,2]) 
#fig.add_traces([go.Scatter(x=Ye,y=Xe),go.Scatter(x=Yn,y=Xn)],rows=[2,2],cols=[2,2])    
#fig.add_trace(go.Figure(data=data, layout=layout),row=2,col=2)  

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=2, z=0)
)

fig.update_layout(height=1100, width=1000)
plot(fig, filename='Nelder-mead.html')








