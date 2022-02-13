import matlab.engine
import numpy as np
import cma
import pickle
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pynmmso import Nmmso
from pynmmso.wrappers import UniformRangeProblem
from pynmmso.listeners import TraceListener
from pyDOE import *
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from joblib import Parallel, delayed
import multiprocessing
#import dill as pickle
eng = matlab.engine.start_matlab()


# Set required paths
root = '/Users/melikedila/Documents/GitHub/'
path= root + r"/BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataDD.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingDD.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataDD']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingDD']

#%%
n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi
gate = gatesm[0]
# Cost for average across thresholds

num_T = 2   # number of thresholds
samples_T = 1
Threshs = lhs(num_T, samples=samples_T)


def neuro1lp_costf(inputparams):
    #print(inputparams[3])
    agg_cost =[]
    gates = gate
    gates = list(gates)
    gates = matlab.double([gates])
    #agg_x.append(inputparams)
    inputparams = list(inputparams)
    for id in range(len(Threshs)):
        inputparams.append(float(Threshs[id][0])) # add thresholds here 
        inputparams.append(float(Threshs[id][1]))
        inp = np.array(inputparams)
        inputparams = matlab.double([inputparams])
        agg_cost.append(eng.getBoolCost_cts_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1))
        inputparams = list(inp)
    #min_val = min(agg_cost)
    #min_index = agg_cost.index(min_val)   
    avr_cost = np.mean(agg_cost)
    return avr_cost





#xvals.append(temp_xvals[min_index]) 

#inputparams = [3.30465379 ,0.45994357 ,7.13104663, 0.35553208, 0.94976593, 1,0]
#neuro1lp_costf(inputparams)
#neuro1lp(inputparams)
def neuro1lp(inputparams):
    i = inputparams
    if i[0] + i [1] < 24:
       cost=neuro1lp_costf(inputparams)
    else:
        dist = i[0] + i [1] - 24
        cost = dist + neuro1lp_costf(inputparams)
    return cost

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


#euclidean_distances((nodes_list[2], nodes_list[3]))


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


#%%

# NM -- Find LO through NM runs

nonzdelt = 0.05
zdelt = 0.00025/0.05
ndim = 3
st = {}
st_s = {}
threshold = 10e-1
permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
#p ={}

rep = 10
#start = {}
init_sol = np.empty([rep,ndim])
for j in range(rep):
    start_time = time.time()
    x , y = [random.uniform(0,24) for i in range(2)]
    while x+y > 24 :
        x , y = [random.uniform(0,24) for i in range(2)]
    z = np.random.uniform(0,12) 
    init_sol[j] = np.array([x,y,z])
    st[j] = init_sol[j]

    # init_simplex = np.zeros((ndim+1, ndim))
    # init_simplex[0] = np.array(init_sols)
    # init_simplex[init_simplex==0] = zdelt
    # init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[0].reshape(-1,1)* step
    # #I[id] = init_simplex[0]
    # OptimizeResult = minimize(neuro1lp, init_simplex[0], method='nelder-mead', 
    #                       options={'initial_simplex': init_simplex, 'xatol': 1e-8, 'disp': False})
    # xvals_s.append(OptimizeResult.x) 
    # fvals_s.append(OptimizeResult.fun)
    # st_s[j] = init_simplex
    # #st[j] = init_simplex
    # print("--- %s seconds ---" % (time.time() - start_time))  
        
par_start_time = time.time()
simplex0 = np.expand_dims(init_sol,axis=1)
simplex_subs = np.expand_dims(init_sol,axis=1)*step + simplex0
init_simplices = np.concatenate((simplex0, simplex_subs), axis=1)


all_args = [(neuro1lp, init_sol[idx], {'method':'nelder-mead', 'options':{'initial_simplex': simplex, 'xatol': 1e-8,'fatol': 1e-13}}) for idx,simplex in enumerate(init_simplices)]
num_cores = min(multiprocessing.cpu_count(), 32)
OptimizeResults = Parallel(n_jobs=num_cores, backend="threading")(delayed(minimize)(*args, **kwargs) for *args, kwargs in all_args)
xvals = [result.x for result in OptimizeResults]
fvals = [result.fun for result in OptimizeResults]
st[j] = init_sol[j]
par_end_time = time.time()
print('Parallel exec time: ', par_end_time - par_start_time)
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
opt = opt[np.argsort(opt[:, 0])]
fvals_list = opt[:,0]
nodes_list = opt[:,1:ndim+1]
st_list = opt[:,ndim+1]
fvals_list = list(fvals_list)
nodes_list = list(nodes_list)
st_list = list(st_list)


#%%
fvals_post_par = fvals_post

xvals_post_par = xvals_post


simplex_subs = init_simplex[0] + (init_simplex[0] * np.expand_dims(permutations, axis=-1)) * step  ## all simplices
simplex0 = np.repeat(np.expand_dims(np.expand_dims(init_simplex[0], axis=0), axis=0), 8, axis=0)    ## beginning of simplices at the beginning
init_simplices = np.concatenate((simplex0, simplex_subs), axis=1)   ##   all simplices together with the first index itself

#%%



########################################################################### artik edge
global_edges = []
nonzdelt = 0.5
step = np.eye(ndim) * nonzdelt
for p in range(len(nodes_list)):
    start_time = time.time()
    xvals_post = []
    fvals_post = []
    init_simplex = np.zeros((ndim+1, ndim))
    init_simplex[0] = nodes_list[p]
    
    
    # init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
    
    # init_simplices = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1) * step
    

    par_start_time = time.time()
    simplex_subs = init_simplex[0] + (init_simplex[0] * np.expand_dims(permutations, axis=-1)) * step
    simplex0 = np.repeat(np.expand_dims(np.expand_dims(init_simplex[0], axis=0), axis=0), 8, axis=0)
    init_simplices = np.concatenate((simplex0, simplex_subs), axis=1)
    
    all_args = [(neuro1lp, init_simplex[0], {'method':'nelder-mead', 'options':{'initial_simplex': simplex, 'xatol': 1e-8,'fatol': 1e-13}}) for simplex in init_simplices]
    num_cores = min(multiprocessing.cpu_count(), 32)
    OptimizeResults = Parallel(n_jobs=num_cores, backend="threading")(delayed(minimize)(*args, **kwargs) for *args, kwargs in all_args)
    xvals_post = [result.x for result in OptimizeResults]
    fvals_post = [result.fun for result in OptimizeResults]
    par_end_time = time.time()
    print('Parallel exec time: ', par_end_time - par_start_time)
 
    
    # seq_start_time = time.time()
    # for i in range(2**ndim):
    #     init_simplex[1:] = init_simplex[0] + init_simplex[0]*permutations[i].reshape(-1,1)* step
    #     OptimizeResult = minimize(neuro1lp, init_simplex[0], method='nelder-mead',
    #                               options={'initial_simplex': init_simplex, 'xatol': 1e-8,'fatol': 1e-13})
    #     xvals_post.append(OptimizeResult.x)
    #     fvals_post.append(OptimizeResult.fun)
    
    # seq_end_time = time.time()
    # print('Sequential exec time: ', seq_end_time - seq_start_time)
    
    
    edge_indices = update_nodes(xvals_post_par, fvals_post_par, fvals_list, nodes_list, threshold)
    uniqe_edge_indices = list(set(edge_indices))
    count = [edge_indices.count(elem) for elem in uniqe_edge_indices] # to ensure edge indices are unique
    local_edges = [(p, idx, count[j]* 1/(2**ndim)) for j,idx in enumerate(uniqe_edge_indices)] #* 1/((2**ndim)*len(nodes_list))
    nm= np.array(local_edges)
    weight = nm[:,2]
    norm_weight = [float(l)/sum(weight) for l in weight]
    for e,w in enumerate(local_edges):
        w = list(w)
        w[2]= norm_weight[e]
        w= tuple(w)
        local_edges[e]=w
        print(local_edges)
    global_edges = global_edges + local_edges
    print("--- %s seconds ---" % (time.time() - start_time))


print('nodes_list length is: ', len(nodes_list))
print('created edges: ', global_edges)
print('fvals: ', fvals_list)




with open("actual_LONs/nodes_list_G0_100s.txt", "wb") as fp:
 pickle.dump(nodes_list, fp)

with open("actual_LONs/global_edges_G0_100s.txt", "wb") as fp:
 pickle.dump(global_edges, fp)

with open("actual_LONs/fvals_list_G0_100s.txt", "wb") as fp:
 pickle.dump(fvals_list, fp)





