# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:51:22 2021

@author: mk633
"""

import numpy as np
import time
import itertools
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from networkx.drawing.nx_agraph import to_agraph

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


###############################################  function

def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)

##################################################NM

nonzdelt =0.05
zdelt = 0.00025
ndim = 2
k = 1
st = {}
threshold = 10e-5
fvals = []
xvals = []

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
    st[j] = init_simplex[0]

st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list]
st_list= []
for i,k in st.items():
    st_list.append(len(k))



######################################################### save

import pickle

with open('C:/Users/mk633/Desktop/code/sixhump_init13', 'wb') as fp:
    pickle.dump(L, fp)
    
    
#######################################################  open saved

with open ('C:/Users/mk633/Desktop/code/sixhump_init13', 'rb') as fp:
    init_pts = pickle.load(fp)    
 
     
################################################################  save edilenin noktasiyla yeniden olusturmak
   
nonzdelt =0.05
zdelt = 0.00025
ndim = 2
st = {}
threshold = 10e-5
fvals = []
xvals = []
step = np.eye(ndim) * nonzdelt
init_simplex = np.zeros((ndim+1, ndim))
random_pick = np.arange(2**ndim)
np.random.shuffle(random_pick)
i = random_pick[0]
for j in range(len(init_pts)):
    init_simplex[0] = init_pts[j]
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[3].reshape(-1,1)* step      
    OptimizeResult = minimize(camel_back, init_simplex[0], method='nelder-mead', 
                              options={'maxfev': 100*200,'return_all': True,'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13, 'disp': True})
    #iterx.append(OptimizeResult.allvecs)
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    st[j] = init_simplex[0]

st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list] 
######################################################## edge for NM

global_edges = [] 
nonzdelt = 0.65
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
        
####################################################  Network olustur

G = nx.MultiDiGraph()
ff= np.sort(fvals_list)

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#4c4cff' #'#0A0AAA' #blue
c2= '#4c4cff'# '#0080ff' #'#38eeff' #0080ff' # '#4c4cff' #'#5050F0'
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

G.add_nodes_from(network_nodes)

for y in range(len(global_edges)):
    G.add_edge(global_edges[y][0],global_edges[y][1],weight= 10*global_edges[y][2]) 

####################################################  Network kaydet

nx.write_graphml(G,'C:/Users/mk633/Desktop/code/NM_sixhump_65_1.graphml')   


#######################################################   BH
k=1
ndim=2
threshold = 10e-5
st = {}
fvals = []
xvals = []
for j in range(len(init_pts)):
    x0 = init_pts[j]
    res = minimize(camel_back, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-07, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
    xvals.append(res.x) 
    fvals.append(res.fun)
    st[j] = x0
    
    
st = optima_merge(xvals, st, threshold)


nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list]

beta = 0.23
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
        uniqe_edge_indices = list(set(edge_indices))
        count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique #bu uniqie edgeler kac tane bulunmus ona bakiyor  
        local_edges = [(p, idx, count[j]* 1/(1000)) for j,idx in enumerate(uniqe_edge_indices) ] #* 1/((2**ndim)*len(nodes_list)) 
        nm= np.array(local_edges)
        nm = nm[:,2]
        norm = [float(i)/sum(nm) for i in nm]
        #if len(local_edges)>0:
        for e,w in enumerate(local_edges):
            w = list(w)
            w[2]= norm[e]
            w= tuple(w)
            local_edges[e]=w
            #print(local_edges) 
    global_edges = global_edges + local_edges



G = nx.MultiDiGraph()
ff= np.sort(fvals_list)

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1= '#4c4cff' # '#0A0AAA' ##0A0AAA #blue
c2= '#4c4cff' #'#5050F0' #'#388bff '#73f3ff' #27a6b2' #32d6e5'#5050F0' #'#38eeff' # #0080ff' # '#4c4cff' #'#5050F0'
n=len(fvals_list)
fvals_list = [round(num, 4) for num in fvals_list]
node_color= ['#FF0000' if(i==min(fvals_list)) else colorFader(c1,c2,i/n) for i in fvals_list]


node_sz = [v.shape[0]*5 for k,v in st.items()]
print(node_color)
posx = [elem[0] for i,elem in enumerate(nodes_list)]
posy = [elem[1] for i,elem in enumerate(nodes_list)]
posz = [fvals_list[i] for i,elem in enumerate(nodes_list)]
#pos = [(ele[0],ele[1]) for ele in ret] 
network_nodes = [(i, {'color':node_color[i],'style':'filled', 'size':node_sz[i] ,'posx': posx[i], 'posy':posy[i], 'posz': posz[i]\
                      }) for i in np.arange(len(nodes_list))]

G.add_nodes_from(network_nodes)

for y in range(len(global_edges)):
    G.add_edge(global_edges[y][0],global_edges[y][1],weight= 10*global_edges[y][2]) 


nx.write_graphml(G,'C:/Users/mk633/Desktop/code/BH_sixhump_44_2.graphml')   

#############################################################  simdi sirada ikisinin basin size'lari grid e plot etme var
import numpy as np

###bh icin siralama
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
st_bh = list(st_list)

## nm icin siralama

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
st_nm = list(st_list)


fig = plt.figure(figsize=(8,8)) ###plt.figure icinde yazmazsan kare cikmaz
#ax = fig.gca()
ax = fig.add_subplot(111)
#ax.set_xticks(np.arange(0, 1, 0.1))
#ax.set_yticks(np.arange(0, 1., 0.1))
plt.scatter(st_nm, st_bh,c='r', s=25)
plt.xlabel("NM")
plt.ylabel("BH")
plt.grid(True)
ax.grid(which='minor', alpha=0.6)
ax.grid(which='major', alpha=0.6)
plt.show()


  
bh_b=[]
nm_b=[]

bh_b.append(st_bh)
nm_b.append(st_nm)

with open('C:/Users/mk633/Desktop/code/nm_basinsizes', 'wb') as fp:
    pickle.dump(nm_b, fp)
    
with open('C:/Users/mk633/Desktop/code/bh_basinsizes', 'wb') as fp:
    pickle.dump(bh_b, fp)    


from statistics import mean,stdev

mean_nm = [float(sum(col))/len(col) for col in zip(*nm_b)]
mean_bh = [float(sum(col))/len(col) for col in zip(*bh_b)]

sd_nm = [float(stdev(col)) for col in zip(*nm_b)]
sd_bh = [float(stdev(col))*0.6 for col in zip(*bh_b)]

fig = plt.figure(figsize=(8,8)) ###plt.figure icinde yazmazsan kare cikmaz
#ax = fig.gca()
ax = fig.add_subplot(111)
#ax.set_xticks(np.arange(0, 1, 0.1))
#ax.set_yticks(np.arange(0, 1., 0.1))
plt.scatter(mean_nm, mean_bh,c='g', s=25)
plt.title('Mean basin size comparison of Nelder-Mead and Basin Hopping')
plt.xlabel("NM")
plt.ylabel("BH")
plt.grid(True)
ax.grid(which='minor', alpha=0.6)
ax.grid(which='major', alpha=0.6)
plt.errorbar(mean_nm, mean_bh, xerr=sd_nm, yerr=sd_bh, fmt='o',ecolor='gray',color='g')
plt.plot([0, 30], [0, 30], color = 'red', linewidth = 2)
plt.show()


se_nm = []
