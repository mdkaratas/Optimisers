#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:29:43 2022

@author: melikedila
"""

import numpy as np
import pickle
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl



###############################################################################  neuro1lp LON data
with open("Desktop/actual_LONs/fvals_list_G0_100s.txt", "rb") as fp:   
    f_vals_0 = pickle.load(fp)   
with open("Desktop/actual_LONs/fvals_list_G1_100s.txt", "rb") as fp:   
    f_vals_1 = pickle.load(fp)  
with open("Desktop/actual_LONs/fvals_list_G2_100s.txt", "rb") as fp:   
    f_vals_2 = pickle.load(fp)   
with open("Desktop/actual_LONs/fvals_list_G3_100s.txt", "rb") as fp:   
    f_vals_3 = pickle.load(fp)       
    
    
with open("Desktop/actual_LONs/global_edges_G0_100s.txt", "rb") as fp:   
    global_edges_0 = pickle.load(fp)   
with open("Desktop/actual_LONs/global_edges_G1_100s.txt", "rb") as fp:   
    global_edges_1  = pickle.load(fp)  
with open("Desktop/actual_LONs/global_edges_G2_100s.txt", "rb") as fp:   
    global_edges_2  = pickle.load(fp)   
with open("Desktop/actual_LONs/global_edges_G3_100s.txt", "rb") as fp:   
    global_edges_3  = pickle.load(fp)        
    
with open("Desktop/actual_LONs/nodes_list_G0_100s.txt", "rb") as fp:   
    nodes_list_G0 = pickle.load(fp)   
with open("Desktop/actual_LONs/nodes_list_G1_100s.txt", "rb") as fp:   
    nodes_list_G1 = pickle.load(fp)  
with open("Desktop/actual_LONs/nodes_list_G2_100s.txt", "rb") as fp:   
    nodes_list_G2 = pickle.load(fp)   
with open("Desktop/actual_LONs/nodes_list_G3_100s.txt", "rb") as fp:   
    nodes_list_G3  = pickle.load(fp)       

    

############################################################################################
############################################################################################  bu !  g1

f_list = f_vals_0
nodes_list = nodes_list_G0
global_edges = global_edges_0


bs = {}
for i in range(len(f_list)):
    #print(i[1])
    c = 0
    for j in global_edges:
        if (j[1] == i):
            c+= 1
    bs[i] = c
    
bs_n = []
for k,v in bs.items():
    bs_n.append(5*v)
posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [f_list[i] for i,elem in enumerate(nodes_list)]

pos = {}
for i in range(len(posx)):
    pos[i] = (posx[i],posz[i])
  
    
edge_width = []
for i in global_edges:
    edge_width.append(0.6*i[2])

G = nx.MultiDiGraph()

network_nodes = [(i, {'style':'filled', 'posx': posx[i], 'posy':posy[i], 'posz': posz[i]\
                      }) for i in np.arange(len(nodes_list))]
#pos = [(elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)]
#pos_dict={}
#for t in range(len(nodes_list)):
    #os_dict[t]= pos[t]
G.add_nodes_from(network_nodes)

for y in range(len(global_edges)):
    G.add_edge(global_edges[y][0],global_edges[y][1],weight= 15*global_edges[y][2]) 
# Create the graph
fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='white') 
plt.axis('off')
#node_color = ['rgb(255,0,0)' if(f==min(f_list)) else 'rgb(0,9,26)' for f in f_list]
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

#c1='#CBC3E3'#9F2B68'#D8BFD8'#953553'#E3735E'#0A0AAA' #blue
#c2= '#DC143C'#953553'#800080'#BA0F30'#880808' ##38eeff' #0080ff' # '#4c4cff' #'#5050F0'
n=len(f_list)
#f = f_list
node_color = ['#770737' if(f==min(f_list)) else '#4169E1' for f in f_list]
#node_color= ['#FF0000' if(i==min(f)) else colorFader(c1,c2,i/n) for i in f]
#print(node_color)
#network_nodes = [(i, {'color':node_color,'style':'filled', \
                      #}) for i in np.arange(len(nodes_list))]
#pos = nx.kamada_kawai_layout(G)  
options = {
    'node_color': node_color,
    'node_size': bs_n,
    'width': edge_width,
    'arrowstyle': '-|>',   ### MI run ettiginde farkli gateler icin kullan
    'edge_color' :'gray','arrowsize': 6,
    'pos' : pos
    #,'edge_curved': 0.2
}



nx.draw_networkx(G,arrows=True, **options,with_labels =False,connectionstyle = 'arc3') #,connectionstyle = 'arc3'
plt.savefig('Desktop/LON_3.eps', format='eps',bbox_inches='tight')






############################################################################################




#############################################################################################    


    
bs = {}
for i in range(len(f_vals_1)):
    #print(i[1])
    c = 0
    for j in global_edges_1:
        if (j[1] == i):
            c+= 1
    bs[i] = c

        

    
    
 
    
 
###############################################################################  3D LONs   g= 01

f = f_vals_1
nodes_list = nodes_list_G1
global_edges  = global_edges_1
G = nx.MultiDiGraph()

ff= np.sort(f)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#0A0AAA' #blue
c2= '#38eeff' #0080ff' # '#4c4cff' #'#5050F0'
n=len(f_vals_1)
node_color= ['#FF0000' if(i==min(f)) else colorFader(c1,c2,i/n) for i in f]


node_sz = [v*5 for k,v in bs.items()]
print(node_color)
posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [elem[2] for i,elem in enumerate(nodes_list)]
#posz = [fvals_list[i] for i,elem in enumerate(nodes_list)]
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
nx.write_graphml(G,'Desktop/neuro1lp_1.graphml')   


######################################################################
    
f_list = f_vals_1
nodes_list = nodes_list_G1
global_edges  = global_edges_1

fig = make_subplots(rows=1, cols=1,specs=[[{"type": "scatter3d"}]], horizontal_spacing = 0.05,vertical_spacing=0.05, subplot_titles=("a) 3-D Landscape visualisation"))



G = nx.MultiDiGraph()

# use indices as node's names
node_color = ['rgb(255,0,0)' if(f==min(f_list)) else 'rgb(0,9,26)' for f in f_list]
#print(node_color)
network_nodes = [(i, {'color':node_color[i],'style':'filled', \
                      }) for i in np.arange(len(nodes_list))]
G.add_nodes_from(network_nodes)
G.add_weighted_edges_from([ (tup[0], tup[1],10*tup[2]) for tup in global_edges])


pos = [(elem[0], elem[1], elem[2]) for i,elem in enumerate(nodes_list)] # fvals_list
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
    
    

    

trace1=go.Scatter3d(x=Xe,     #####  edge trace
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(0,0,255)', width=2),  #'rgb(125,125,125)'
               hoverinfo='none'
               )

node_sz = [v*3+1 for k,v in bs.items()] 
# create nodes 3D trace
trace2=go.Scatter3d(x=Xn,       #####  node trace
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(175,0,42)', width=1)  #()
                             ),
               text=[str(n) for n in G.nodes],
               hoverinfo='text'
               )    
data=[trace1, trace2]
fig.add_traces([go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',
               line=dict(color='gray', width=2)),go.Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',   ### edges
               name='actors',
               marker=dict(symbol='circle',
                             size=node_sz,
                             color=node_color,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=1)  ## around nodes
                             ),
               text=[str(n) for n in G.nodes])],rows=[1,1],cols=[1,1])   
                                                                          
                                                                          
camera = dict(
    up=dict(x=1, y=1, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=2, z=0)
)

fig.update_layout(height=1100, width=900, title_text="Subplots with Annotations")
plot(fig, filename='neuro1lp_g1.html')                                                                          

import igraph as ig

Gix = ig.read('neuro1lp_1.graphml',format="graphml")




G = nx.MultiDiGraph()

node_sz = [v*5 for k,v in bs.items()]
print(node_color)
posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [elem[2] for i,elem in enumerate(nodes_list)]
#posz = [fvals_list[i] for i,elem in enumerate(nodes_list)]
#pos = [(ele[0],ele[1]) for ele in ret] 
network_nodes = [(i, {'color':node_color[i],'style':'filled', 'size':node_sz[i] ,'posx': posx[i], 'posy':posy[i], 'posz': posz[i]\
                      }) for i in np.arange(len(nodes_list))]
#pos = [(elem[1], fvals_list[i]) for i,elem in enumerate(nodes_list)]
#pos_dict={}
#for t in range(len(nodes_list)):
    #os_dict[t]= pos[t]
G.add_nodes_from(network_nodes)

for y in range(len(global_edges)):
    G.add_edge(global_edges[y][0],global_edges[y][1],weight= 5*global_edges[y][2]) 
fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='white')     
nx.draw_networkx(G, arrows=True,**options, with_labels =False)    