#-*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:42:21 2020

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
            if not merged_bool[i]:  
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
ndim = 3
n = 1
k = 1
st = {}
threshold = 10e-5
fvals = []
xvals = []

#permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
#perm[perm==0] = -1
permutations = np.array(list(itertools.product(range(2), repeat=3)))
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 100 
L = []
for j in range(rep):
    x0_cam_0 = np.random.uniform(low=-3, high=3, size=(k, 1))
    x0_cam_1 = np.random.uniform(low=-2, high=2, size=(k, 1))
    x0_cam_2 = np.random.uniform(low=-2, high=2, size=(k, 1))
    init_simplex = np.zeros((ndim+1, ndim))
    random_pick = np.arange(2**ndim)
    np.random.shuffle(random_pick)
    i = random_pick[0]
    init_simplex[0] = np.array([x0_cam_0, x0_cam_1,x0_cam_2]).ravel()
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[0].reshape(-1,1)* step
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

      
import pickle

with open('C:/Users/mk633/Desktop/code/L_sixhump', 'wb') as fp:
    pickle.dump(L, fp)


#################################### the same initial points to search
#%%
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph


import pickle
with open ('C:/Users/mk633/Desktop/code/L_sixhump', 'rb') as fp:
    L= pickle.load(fp)
 
def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)


permutations = np.array(list(itertools.product(range(2), repeat=2)))
permutations[permutations==0] = -1    
nonzdelt =0.05
zdelt = 0.00025
ndim = 2
k=1
n=1
st = {}
threshold = 10e-5
fvals = []
xvals = []
step = np.eye(ndim) * nonzdelt
init_simplex = np.zeros((ndim+1, ndim))
random_pick = np.arange(2**ndim)
np.random.shuffle(random_pick)
i = random_pick[0]
for j in range(len(L)):
    init_simplex[0] = L[j]
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[2].reshape(-1,1)* step      
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
opt = np.column_stack((fvals_list, nodes_list))
opt = opt[np.argsort(opt[:, 1])]
fvals_list = opt[:,0]
nodes_list = opt[:,1:]
#%%
########################################################################### artik edge
global_edges = [] 
nonzdelt = 0.8
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

    
A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('sutcamel.png')



##########################################################################
# draw the network

plt.figure(num=None, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')

G = nx.MultiDiGraph()

# use indices as node's names
#node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
#print(node_color)
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#0A0AAA' #blue
c2= '#38eeff' #0080ff' # '#4c4cff' #'#5050F0'
n=len(fvals_list)
node_color= ['#FF0000' if(i==min(fvals_list)) else colorFader(c1,c2,i/n) for i in fvals_list]

network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from([ (tup[0], tup[1],tup[2]) for tup in global_edges])



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
trace1=go.Scatter3d(x=Ye,
               y=Xe,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0] for k,v in st.items()]  ########

# create nodes 3D trace
trace2=go.Scatter3d(x=Yn,
               y=Xn,
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


data=[trace1, trace2]
x_interval = (-1.9,1.9)
y_interval = (-1.1, 1.1)
x = np.linspace(x_interval[0], x_interval[1], 100)
y = np.linspace(y_interval[0], y_interval[1], 100)
x,y = np.meshgrid(x,y)
z= (4-2.1*x**2 + x**4 /3)*x**2 + x*y + (4*y**2 -4)*y**2
fig = go.Figure(data=[go.Surface(z=z, x=y, y=x)],layout=layout)
fig=go.Figure(data=data, layout=layout)


camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=2, z=0)
)

plot(fig, filename='Nelder-mead.html')

##########################################yanyana dortlu cizme plotlari
    
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2)





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
trace1=go.Scatter3d(x=Ye,
               y=Xe,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=2),
               hoverinfo='none'
               )

node_sz = [v.shape[0] for k,v in st.items()]  ########

# create nodes 3D trace
trace2=go.Scatter3d(x=Yn,
               y=Xn,
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


data=[trace1, trace2]
fig.add_trace(go.Figure(data=data, layout=layout), row=1, col=2)

fig.update_layout(
    title_text='3D subplots with different colorscales',
    height=800,
    width=800
)
plot(fig, filename='Nelder-mead.html')
############################


# use fitness values as node's names
#fvals_round = np.round(fvals_list, 3)
#G.add_nodes_from(fvals_round)
#G.add_edges_from([[fvals_round[tup[0]], fvals_round[tup[1]]] for tup in global_edges])


# rename nodes
#fvals_round = list(np.round(fvals_list, 3))
#print(fvals_round)
#fvals_names = []
#charac = 97
#for v in fvals_round:
    #fvals_names.append(str(v)+'_%s'%(chr(charac)))
    #charac+=1
    
# = {k:fvals_names[k] for k in range(len(nodes_list)) }
#print('mapping', mapping)
#nx.relabel_nodes(G, mapping, copy=False)

#nx.draw_networkx(G, with_label = True, node_size=node_sz, width=[10*tup[2] for tup in global_edges], \
                 #node_color =node_color, connectionstyle='arc3, rad = 0.7')
    
#A = to_agraph(G) 
#A.layout('dot')                                                                 
#A.draw('cameles.png')
#%%
########################## MERGING FOR CM-LON \\ FUNNELS


#%%
########################## MERGING FOR CM-LON \\ FUNNELS


edge_lst = [list(elem) for elem in global_edges]
upfvals_list= list(fvals_list)

import pandas as pd
df = pd.DataFrame(upfvals_list)
df
## edge index change + upfvals_list fazla nodelu olanlari aynilarini attik
index = []
for i in range(len(fvals_list)-1):
    #print(i)
    for j in np.arange(i+1,len(fvals_list)):
        #print(j)
        if (fvals_list[i]==fvals_list[j]):
            index.append(j)
            for k in edge_lst:
                #print(k[0])
                if (k[0] == j):
                    k[0]=i
                if (k[1] == j):
                    k[1]=i
                upfvals_list = df.drop(df.index[index])  

## aggregarted edges
index_edge = []
                  
for k in range(len(edge_lst)-1):
    for j in np.arange(k+1,len(edge_lst)):
       #print(j)
       if (edge_lst[k][0] == edge_lst[j][0] and edge_lst[k][1] == edge_lst[j][1]):
           edge_lst[k][2]+=edge_lst[j][2]
           edge_lst[j][2]= edge_lst[k][2]
           index_edge.append(j)
upedge_list = pd.DataFrame(edge_lst)            
upedge_list = upedge_list.drop(upedge_list.index[index_edge])  





















