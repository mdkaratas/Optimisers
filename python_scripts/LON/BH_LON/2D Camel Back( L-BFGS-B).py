# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:04:20 2020

@author: mk633
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import time
#from networkx.drawing.nx_agraph import to_agraph

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


def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)

k=1
n=1
ndim = 2
threshold = 10e-5
nonzdelt = 0.05
zdelt = 0.00025


permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt

st = {}
fvals = []
xvals = []
rep = 100
for j in range(rep):
    x0_cam_0 = np.random.uniform(low=-1.9, high=1.9, size=(k, n))
    x0_cam_1 = np.random.uniform(low=-1.1, high=1.1, size=(k, n))
    x0 = np.array([x0_cam_0, x0_cam_1]).ravel()
    res = minimize(camel_back, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
    xvals.append(res.x) 
    fvals.append(res.fun)
    st[j] = x0
    
    
st = optima_merge(xvals, st, threshold)


nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list]
opt = np.column_stack((fvals_list, nodes_list))
opt = opt[np.argsort(opt[:, 1])]
fvals_list = opt[:,0]
nodes_list = opt[:,1:]

beta = 0.4546
global_edges = []


for p in range(len(nodes_list)): 
    start_time = time.time()
    xvals_post = []
    fvals_post = []
    x0 = nodes_list[p]
    r = 0
    while r< 1000: #for i in range(1100): 
        shake= np.random.uniform(low=-0.5*beta, high=0.5*beta, size=(k, ndim))
        x0 = x0 + shake
        res = minimize(camel_back, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
        xvals_post.append(res.x) 
        fvals_post.append(res.fun)
        edge_indices = [] #bulunanlarin uzakliklariyla nodes_list icinden hangisine match ettigini veriyor,unutma nodes_listin birinden pert yaptik yenileri hep aynisina mi giidyor diye bakiyor
        dist = euclidean_distances(xvals_post, nodes_list) #edge_indices = update_nodes(xvals_post, fvals_post, fvals_list, nodes_list, threshold)
        dist_bool = dist<=10e-5
        for i, row in enumerate(dist_bool):
            ind = np.where(row)[0]
            if(ind.size==0):
                #print(xvals_post[i],fvals_post[i],i)
                nodes_list.append(xvals_post[i])
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
        nm = np.array(local_edges)
        nm = nm[:,2]
        norm = [float(i)/sum(nm) for i in nm]
        local_edges = list(local_edges)
        #local_edges[:][2]
        #norm_local_edges = [(p, idx, norm[t] for t ) for j,idx in enumerate(uniqe_edge_indices) ] 
    global_edges = global_edges + local_edges
    
    print("--- %s seconds ---" % (time.time() - start_time))    
   
print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)
#%%
import matplotlib as mpl
from networkx.drawing.nx_agraph import to_agraph
G = nx.MultiDiGraph()
ff= np.sort(fvals_list)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#0A0AAA' #blue
c2= '#0A0AAA'  # 0080ff'  #38eeff' #0080ff' # '#4c4cff' #'#5050F0'
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
nx.write_graphml(G,'lbfgsb.graphml')   


#%%
from networkx.drawing.nx_agraph import to_agraph   
G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
node_sz = [v.shape[0]*5 for k,v in st.items()]
print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])
#mapping = {k:fvals_list[k] for k in range(len(nodes_list)) }
#print('mapping', mapping)
#nx.relabel_nodes(G, mapping, copy=False)

plt.figure(figsize =(30, 30))
pos = nx.circular_layout(G) 
nx.draw_networkx(G, pos,with_label = True, node_size=node_sz,node_color =node_color, width=[2 *tup[2] for tup in global_edges], connectionstyle='arc3', rad = '0.3') 

nx.draw_networkx(G, with_label = True, node_size=node_sz, width=[tup[2] for tup in global_edges], connectionstyle='arc3', rad = '0.3')
    
A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('cam_benchmark.png')   

 


# draw the network

plt.figure(num=None, figsize=(30, 30), dpi=80, facecolor='w', edgecolor='k')

G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['red' if(f==min(fvals_list)) else 'blue' for f in fvals_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_edges_from([ (tup[0], tup[1]) for tup in global_edges])



#       from chart_studio import plotly as py
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
         title="Downhill Simplex- Camelback Benchmark Function",
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


plot(fig, filename='L-BFGS-B.html')











     