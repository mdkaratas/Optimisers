# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:49:50 2020

@author: mk633
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm


def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)
nonzdelt = 0.05
zdelt = 0.00025/0.05
N = 2
n = 1
k = 1

one2np1 = list(range(1, N + 1))
fsim = np.empty((N + 1,), float)
np.random.seed(9)
x0_cam_0 = np.random.uniform(low=-1.9, high=1.9, size=(k, n))
x0_cam_1 = np.random.uniform(low=-1.1, high=1.1, size=(k, n))
permutations = (( np.arange(2**N).reshape(-1, 1) & (2**np.arange(N))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(N) * nonzdelt

sim = np.zeros((N+1, N))
random_pick = np.arange(2**n)
np.random.shuffle(random_pick)
i = random_pick[0]
sim[0] = np.array([x0_cam_0, x0_cam_1]).ravel() 
sim[sim==0] = zdelt
sim[1:] = sim[0] +  sim[0]*permutations[i].reshape(-1,1)* step
OptimizeResult = minimize(camel_back, sim[0], method='nelder-mead', 
                          #options={'maxfev': 1000*2000, 'return_all': True, 'initial_simplex': sim, 'fatol': 1e-18, 'disp': True})



#%%
allvecs = [sim]
N = 2
rho = 1
chi = 2
psi = 0.5
sigma = 0.5

maxiter = 200
iterations = 0
while (iterations < maxiter):

    xbar = np.add.reduce(sim[:-1], 0) / N
    xr = (1 + rho) * xbar - rho * sim[-1]
    fxr = camel_back(xr)
    doshrink = 0
    
    if fxr < fsim[0]:
        xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
        fxe = camel_back(xe)
    
        if fxe < fxr:
            sim[-1] = xe
            fsim[-1] = fxe
        else:
            sim[-1] = xr
            fsim[-1] = fxr
    else:  # fsim[0] <= fxr
        if fxr < fsim[-2]:
            sim[-1] = xr
            fsim[-1] = fxr
        else:  # fxr >= fsim[-2]
            # Perform contraction
            if fxr < fsim[-1]:
                xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                fxc = camel_back(xc)
    
                if fxc <= fxr:
                    sim[-1] = xc
                    fsim[-1] = fxc
                else:
                    doshrink = 1
            else:
                # Perform an inside contraction
                xcc = (1 - psi) * xbar + psi * sim[-1]
                fxcc = camel_back(xcc)
    
                if fxcc < fsim[-1]: 
                    sim[-1] = xcc
                    fsim[-1] = fxcc
                else:
                    doshrink = 1
    
            if doshrink:
                for j in one2np1:
                    sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                    fsim[j] = camel_back(sim[j])
    ind = np.argsort(fsim)
    sim = np.take(sim, ind, 0)
    fsim = np.take(fsim, ind, 0)
    allvecs.append(sim)
    iterations += 1        
       
    
    
    
    simplices = np.array(allvecs[0:20])
    
    
    abc = OptimizeResult.allvecs
    
    x_interval = (-1.9,1.9)
    y_interval = (-1.1, 1.1)
    
    xlist = np.linspace(x_interval[0], x_interval[1], 100)
    ylist = np.linspace(y_interval[0], y_interval[1], 100)
    X, Y = np.meshgrid(xlist, ylist)
    def camel_back(x,y):
        return ( (4-2.1*x**2 + x**4 /3)*x**2 + x*y + (4*y**2 -4)*y**2)
    Z = camel_back(X, Y)   
    
    fig,ax=plt.subplots(1,1,figsize=(10, 10))
    cp = ax.contourf(X, Y, Z,35)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    #ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    #for i in range(simplices.shape[]):
    #cmap = cm.get_cmap('PiYG', 19)       
    for i in range(len(abc)): 
        rand_hex_color = '#%06x'% (np.random.randint(0,16**6))
        ax.scatter(abc[i][0],abc[i][1], c=rand_hex_color)
    
        
    #ax.legend(['%02d'%(i) for i in range(len(abc))])
    #ax.plot(simplices[1,:,0], simplices[1,:,1], 'bo')
    #ax.plot(simplices[2,:,0], simplices[2,:,1], 'bo')
    plt.show()


simplices[1:][0][1]

simplices.shape[0]


simplices[i,:,0]


simplices[0,:,0]
simplices[0,0,0]

r = np.hstack((simplices[i,:,0],simplices[i,0,0]))
#%%
 orjinal hali


nonzdelt = 0.05
zdelt = 0.00025/0.05
ndim = 2
n = 1
k = 1
st = {}
threshold = 10e-5
fvals = []
xvals = []

permutations = (( np.arange(2**ndim).reshape(-1, 1) & (2**np.arange(ndim))) != 0).astype(int)
permutations[permutations==0] = -1 
step = np.eye(ndim) * nonzdelt
rep = 300
hadiw = []
np.random.seed(0)  
for j in range(rep):
    x0_cam_0 = np.random.uniform(low=-1.9, high=1.9, size=(k, n))
    x0_cam_1 = np.random.uniform(low=-1.1, high=1.1, size=(k, n))
    init_simplex = np.zeros((ndim+1, ndim))
    random_pick = np.arange(2**n)
    np.random.shuffle(random_pick)
    i = random_pick[0]
    init_simplex[0] = np.array([x0_cam_0, x0_cam_1]).ravel()    
    init_simplex[init_simplex==0] = zdelt
    init_simplex[1:] = init_simplex[0] +  init_simplex[0]*permutations[i].reshape(-1,1)* step
    OptimizeResult = minimize(camel_back, init_simplex[0], method='nelder-mead', 
                              options={'return_all': True,'maxfev': 10*200,'initial_simplex': init_simplex, 'fatol': 1e-18, 'disp': False})
    hadiw.append(OptimizeResult.allvecs)
    xvals.append(OptimizeResult.x) 
    fvals.append(OptimizeResult.fun)
    st[j] = init_simplex
    
st = optima_merge(xvals, st, threshold)     
#print('st: ', st)

# create the initial list of nodes
nodes_list = [xvals[k] for k,v in st.items()]
fvals_list = [fvals[k] for k,v in st.items()]
fvals_list = [round(num, 12) for num in fvals_list]
#print(nodes_list)   