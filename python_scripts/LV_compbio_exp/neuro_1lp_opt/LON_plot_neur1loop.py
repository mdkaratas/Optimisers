#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:07:46 2022

@author: mkaratas
"""

###  LON for 1 loop Neurospora


import numpy as np
import pickle
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA


###############################################################################  Arabid2lp LON data

read_root = "/Users/mkaratas/Desktop/GitHub/Optimisers/Llyonesse/continuous_fcn_results/LON_data/neur_1lp/ME/"


with open(read_root + "fvals_list_0.txt", "rb") as fp:   
    f_0 = pickle.load(fp)   
with open(read_root + "fvals_list_1.txt", "rb") as fp:   
    f_1 = pickle.load(fp)  
with open(read_root + "fvals_list_2.txt", "rb") as fp:   
    f_2 = pickle.load(fp)     
with open(read_root + "fvals_list_3.txt", "rb") as fp:   
    f_3 = pickle.load(fp)     
    
    
with open(read_root + "global_edges_0.txt", "rb") as fp:   
    global_edges_0 = pickle.load(fp)   
with open(read_root + "global_edges_1.txt", "rb") as fp:   
    global_edges_1 = pickle.load(fp)  
with open(read_root + "global_edges_2.txt", "rb") as fp:   
    global_edges_2  = pickle.load(fp)   
with open(read_root + "global_edges_3.txt", "rb") as fp:   
    global_edges_3 = pickle.load(fp)   
    
    
with open(read_root + "nodes_list_0.txt", "rb") as fp:   
    nodes_list_0 = pickle.load(fp)   
with open(read_root + "nodes_list_1.txt", "rb") as fp:   
    nodes_list_1 = pickle.load(fp)  
with open(read_root + "nodes_list_2.txt", "rb") as fp:   
    nodes_list_2 = pickle.load(fp)   
with open(read_root + "nodes_list_3.txt", "rb") as fp:   
    nodes_list_3 = pickle.load(fp)   

with open(read_root + "nodes_list_0.txt", "rb") as fp:   
    st_list_0 = pickle.load(fp)   
with open(read_root + "nodes_list_1.txt", "rb") as fp:   
    st_list_1 = pickle.load(fp)  
with open(read_root + "nodes_list_2.txt", "rb") as fp:   
    st_list_2 = pickle.load(fp)   
with open(read_root + "nodes_list_3.txt", "rb") as fp:   
    st_list_3 = pickle.load(fp)   
    
#################################################
################################################
###############################################


st_list = st_list_0
nodes_list = nodes_list_0
f_list = f_0
global_edges = global_edges_0

############  PCA for x axes layout

pca = PCA(1)  # project from 64 to 2 dimensions
projected_G0 = pca.fit_transform(nodes_list)
list_of_floats = [float(item) for item in projected_G0]



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
    bs_n.append(12*v+1)



posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [f_list[i] for i,elem in enumerate(nodes_list)]


pos = {}
for i in range(len(posx)):
    pos[i] = (list_of_floats[i],posz[i])
      
    
edge_width = []
for i in global_edges:
    edge_width.append(3*i[2])

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
#fig = plt.figure(figsize=(3,3),dpi=5000,facecolor='white') 
fig, ax = plt.subplots(figsize=(8, 5),dpi=1000,facecolor='white')
ax.set_facecolor("white")
ax.set_axis_on()
ax.set_ylim([0,4])
plt.grid(False)
#plt.axis('on')
#node_color = ['rgb(255,0,0)' if(f==min(f_list)) else 'rgb(0,9,26)' for f in f_list]
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

#c1='#CBC3E3'#9F2B68'#D8BFD8'#953553'#E3735E'#0A0AAA' #blue
#c2= '#DC143C'#953553'#800080'#BA0F30'#880808' ##38eeff' #0080ff' # '#4c4cff' #'#5050F0'
n=len(f_list)
#f = f_list  #A52A2A    #770737
node_color = ['#8B0000'  if(f==min(f_list)) else '#E9967A' for f in f_list]
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
    ,'ax':ax
    #,'edge_curved': 0.2
}



nx.draw_networkx(G,arrows=True, **options,with_labels =False,connectionstyle = 'arc3') #,connectionstyle = 'arc3'

ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.rcParams["figure.autolayout"] = True
plt.savefig('Desktop/cont_figs/LON_ME_0.eps', format='eps',bbox_inches='tight')



