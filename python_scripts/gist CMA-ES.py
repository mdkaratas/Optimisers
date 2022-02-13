# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:22:44 2021

@author: mk633
"""
import cma

with Parallel(n_jobs=n_jobs, backend='loky') as pool:
            
    # feas function - returns a function that takes in (x, f)
    # with a default value of f
    ff = is_feasible(bound_arrays, bound_num, lb, ub)
    
    # set up CMA-ES on the selected variables
    n_dim = lb.size
    
    cma_lb = np.zeros((n_dim), dtype='float')
    cma_ub = #np.ones((n_dim), dtype='float')
    
    cma_options = {'bounds':[list(cma_lb), list(cma_ub)], 
                   'tolfun':1e-7, 
                   'maxfevals': max_fevals,
                   'verb_log': 0,
                   'verb_disp': 1, # print every iteration
                   'CMA_stds': np.abs(cma_ub - cma_lb),
                   'popsize': n_pop,
                   'is_feasible': ff}
    
    cma_sigma =  0.25
    cma_centroid = initial_point_generator(n_dim, ff)
    
    # history storage
    fname = f'H02b_CMAES_BIPOP_{max_fevals:d}_{house_name:s}_{run:d}.npz'
    save_path = os.path.join(results_path, fname)
    
    hs = CMAES_history_storer(n_dim, GAD, CAD, IAD, VAD, bounds, GT, 
                              save_path, save_every_n_calls=10)
    
    print('Starting CMA-ES')
    res = cma.fmin(None, # no function as specifying a parallel one
                   cma_centroid, # random evaluation within bounds
                   cma_sigma,
                   options=cma_options,
                   parallel_objective=par_func(f, lb, ub, pool, hs),
                   bipop=True, 
                   restarts=9)
    
    
    