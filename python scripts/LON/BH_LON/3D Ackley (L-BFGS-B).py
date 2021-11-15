# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:32:55 2020

@author: mk633
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import time

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
           edge_indices.append(len(nodes_list)-1)
        # if this LO was already found and strored inside nodes list
        else:
           edge_indices.append(idx[0])
           
    return edge_indices

def ackley(x):
    return -20* np.exp(-0.2 * np.sqrt( (1/x.shape[0]) * np.sum(x**2) ) ) - \
            np.exp( (1/x.shape[0]) * np.sum( np.cos(2*np.pi*x)) ) + 20 + np.e
            
            
ndim = 3
n = 3
k = 1
threshold = 10e-5


st = {}
fvals = []
xvals = []
rep = 100

import pickle
with open ('C:/Users/mk633/Desktop/code/start_ackley', 'rb') as fp:
    L= pickle.load(fp)

for j in range(len(L)):
    x0 = L[j] #x0 = np.random.uniform(low=-32.768, high=32.768, size=(k, n)) 
    res = minimize(ackley, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-13, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 20000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
    xvals.append(res.x) 
    fvals.append(res.fun)
    st[j] = x0
    
    
st = optima_merge(xvals, st, threshold)



nodes_lista = [xvals[k] for k,v in st.items()]
fvals_lista = [fvals[k] for k,v in st.items()]
fvals_lista = [round(num, 12) for num in fvals_lista]
st_list= []
for i,k in st.items():
    st_list.append(len(k))
opt = np.column_stack((fvals_lista, nodes_lista,st_list))
opt = opt[np.argsort(opt[:, 1])]
fvals_lista = opt[:,0]
nodes_lista = opt[:,1:4]
st_list = opt[:,4]
fvals_lista = list(fvals_lista)
nodes_lista = list(nodes_lista)
st_list = list(st_list)

#bh=[]
#bh.append(4980)
#%%
import pickle
with open ('C:/Users/mk633/Desktop/code/L_ackley', 'rb') as fp:
    L= pickle.load(fp)
beta = 0.4546
import pickle
with open ('C:/Users/mk633/Desktop/code/mutualLO', 'rb') as fp:
    mutual_lo= pickle.load(fp)

with open ('C:/Users/mk633/Desktop/code/mutualf', 'rb') as fp:
    fvals_list= pickle.load(fp)
#%%    
global_edges = []

k=1
ndim= 3

#fvals_list.append(19.1726)

#nodes_list= mutual_lo
for p in range(len(mutual_lo)): 
    #start_time = time.time()
    xvals_post = []
    fvals_post = []
    x0 = mutual_lo[p]
    r = 0
    while r<= 1000: #for i in range(1100): 
        shake= np.random.uniform(low=-0.5*beta, high=0.5*beta, size=(k, ndim))#low=-0.5*beta, high=0.5*beta, size=(k, ndim))
        x0 = x0 + shake
        res = minimize(ackley, x0, args=(), method='L-BFGS-B', jac=None, bounds=((-32.768,32.768),(-32.768,32.768),(-32.768,32.768)), tol=None, callback=None, options={'disp': True, 'maxcor': 10, 'ftol': 1e-13, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 20000,  'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
        xvals_post.append(res.x)  #'maxiter': 15000,
        fvals_post.append(res.fun)
        edge_indices = [] #bulunanlarin uzakliklariyla nodes_list icinden hangisine match ettigini veriyor,unutma nodes_listin birinden pert yaptik yenileri hep aynisina mi giidyor diye bakiyor
        dist = euclidean_distances(xvals_post, mutual_lo) #edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
        dist_bool = dist<=10e-5
        for i, row in enumerate(dist_bool):
            ind = np.where(row)[0]
            if(ind.size==0):
                #print(xvals_post[i],fvals_post[i],i)
                mutual_lo.append(xvals_post[i])
                fvals_list.append(fvals_post[i])
                
                edge_indices.append(len(nodes_list)-1)
            else: 
                r = r+1
                if fvals_list[int(ind[0])]<= fvals_list[p]:
                    edge_indices.append(ind[0])
                    
        # gittigi edgeler aslinda hangileri her iterationda farkli gittigini veriyordu edge-#ind simdi unique edgele hangilerine gittigini verecek
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique #bu uniqie edgeler kac tane bulunmus ona bakiyor  
    local_edges = [(p, idx, count[j]* 1/(1000)) for j,idx in enumerate(uniqe_edge_indices) ] #* 1/((2**ndim)*len(nodes_list)) 
    #nm = np.array(local_edges)
    #nm = nm[:,2]
    #norm = [float(i)/sum(nm) for i in nm]
    #local_edges = list(local_edges)
    #local_edges[:][2]
    #norm_local_edges = [(p, idx, norm[t] for t ) for j,idx in enumerate(uniqe_edge_indices) ] 
    global_edges = global_edges + local_edges
    
    #print("--- %s seconds ---" % (time.time() - start_time))  
        
   
print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)
   
print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)

G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'orange' for f in fvals_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from(global_edges, weight = 'weight')
#G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])



############
from networkx.drawing.nx_agraph import to_agraph   
G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
#node_color = [G.degree(v) for v in G]
node_sz = [9*fvals_list[k] for k,v in st.items()]
print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])
G.selfloop_edges(data=True)
#G.nodes_with_selfloops()
#G.degree(weight='weight')
#mapping = {k:fvals_list[k] for k in range(len(nodes_list)) }
#print('mapping', mapping)
#nx.relabel_nodes(G, mapping, copy=False)

plt.figure(figsize =(15, 15))
pos = [(elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)]    #node_size=node_sz,
nx.draw_networkx(G, pos,with_label = False, node_color =node_color, width=[2000*tup[2] for tup in global_edges],edge_color ='.4', cmap = plt.cm.Blues, connectionstyle='arc3', rad = '0.3') 

    
A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('ackley.png')  



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

node_sz = [v.shape[0]*10 for k,v in st.items()]  ########

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
         title="Iterated Downhill Simplex- Ackley Benchmark Function",
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


#from plotly.subplots import make_subplots

#fig = make_subplots(rows=1, cols=2)

#fig.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
#fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)

#fig.show()
nnodes_lista= []
for j in range(len(nodes_lista)):
    nnodes_lista.append([round(num, 5) for num in nodes_lista[j][0:3]])

mutual_lo =[]
for i in range(len(nodes_list)):
    for j in range(len(nodes_lista)):
        if (list(nodes_list[i][0]) == list(nodes_lista[j][0])):
            if (list(nodes_list[i][1]) == list(nodes_lista[j][1])):
                if (list(nodes_list[i][2]) == list(nodes_lista[j][2])):
                    mutual_lo.append(list(nodes_list[i][0:3]))
        
        
1,3,7,8,11,16 ,19,23,25,37,44,46,56,59,74,83,84,86,87,91,93,94,95
mutual_lo.append(nodes_list[95])       
import pickle

with open('C:/Users/mk633/Desktop/code/mutualLO', 'wb') as fp:
    pickle.dump(mutual_lo, fp)
    
with open('C:/Users/mk633/Desktop/code/start_ackley', 'wb') as fp:
    pickle.dump(L, fp)
        

fvals_list=[]

19.65710,19.9228,19.90763,19.17262,18.85579,19.84121,19.08367,19.44632,18.47531,17.03045,14.74041,19.44735,18.9747,19.63730,19.759583,19.65989,19.66431,19.30005,19.40987,19.34665,19.43801
fvals_list.append(19.43801)
with open('C:/Users/mk633/Desktop/code/mutualf', 'wb') as fp:
    pickle.dump(fvals_list, fp)











            