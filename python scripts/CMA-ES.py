# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:30:30 2021

@author: mk633
"""
import cma
import numpy as np 
import matplotlib.pyplot 
#help(cma)
#help(cma.fmin)
def camel_back(x):
    return ( -1 *( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2))
cma.CMAOptions()   # returns all the options

lb= np.array([-3,-2])
ub=np.array([3,2])

n_pop= 6

 # set up CMA-ES on the selected variables
n_dim = lb.size
max_fevals= 200*n_dim
#x0=np.array( [0.3])

cma_lb = np.array([-3,-2], dtype='float')
cma_ub = np.array([3,2], dtype='float')


x0= np.array([0.2,0.1])
def is_feasible(x0, f):
    lb = (-3,-2)
    ub = (3,2)
    n= 2
    for t in range(n-1):
        for u in range(n-1):
            while (x0[u,t] < lb[t] or x0[u,t] > ub[t]):
                x0 = np.random.uniform(low=lb[t], high=ub[t], size=(2, 1)) 
    return x0


#ff = (x,f)

cma_options = {'boundary_handling': 'BoundPenalty',
               'bounds':[[-3,-2], [3,2]], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb), # multipliers for sigma0 in each coordinate, not represented in C, makes scaling_of_variables obsolete
                   'popsize': n_pop}#,
                   #'is_feasible': is_feasible(x0,camel_back)}


# sigma0 should be about 1/4th of the search domain width
# BIPOP is a special restart strategy switching between two population sizings 
#- small (like the default CMA, but with more focused search) and
# large (progressively increased as in IPOP). 
#This makes the algorithm perform well both on functions with many regularly or irregularly arranged local optima 
#(the latter by frequently restarting with small populations). 
# For the `bipop` parameter to actually take effect, also select non-zero number of (IPOP) restarts; 
# the recommended setting is ``restarts<=9``
# and `x0` passed as a string using `numpy.rand` to generate initial solutions.
#  Note that small-population restarts do not count into the total restart count.

 # fmin(objective_function, x0, sigma0, options=None, args=(), gradf=None, restarts=0, restart_from_best='False', incpopsize=2, eval_initial_x=False, parallel_objective=None, noise_handler=None, noise_change_sigma_exponent=1, noise_kappa_exponent=0, bipop=False, callback=None)
 
res = cma.fmin(cma.ff.rastrigin, lambda : 2. * np.random.rand(3) - 1, 0.5, options=cma_options, bipop=True, restarts=9) 
res = cma.fmin(camel_back, x0 , 0.5, options=cma_options, bipop=True, restarts=10)
cma.plot();
cma.s.figshow()
