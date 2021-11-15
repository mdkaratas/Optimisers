#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: melikedila
"""

import cma
import numpy as np

def camel_back(x):
    return ( ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2))

cma_lb = np.array([-3,-2], dtype='float')
cma_ub = np.array([3,2], dtype='float')

max_fevals = 10**5
n_pop = 100
 
cma_options = {'bounds':[cma_lb, cma_ub], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb),
                   'popsize': n_pop}#'is_feasible': ff}
init_sol = 2 * [0]
init_sigma = 0.5
es = cma.CMAEvolutionStrategy(init_sol ,init_sigma, cma_options) # optimizes the 2-dimensional function with initial solution all zeros and initial sigma = 0.5

sol = es.optimize(neuro1lp_costf).result   
            