#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:54:12 2022

@author: melikedila
"""
import matlab.engine
import numpy as np
import sys
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
import itertools
import torch
import gpytorch

eng = matlab.engine.start_matlab()

n = 8
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi
#  lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
#gate = gates[gate-1]

#gate1 = sys.argv[1]
#gate2 = sys.argv[2]
#gate3 = sys.argv[3]
#gate4 = sys.argv[4]
#gate5 = sys.argv[5]

#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataLL.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingLL.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataLL']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingLL']


def arabid2lp_gates(inputparams):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = gate
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_arabid2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

def arabid2lp(inputparams):
    for i in inputparams:
        dist = []
        if (inputparams[0] + inputparams[1] + inputparams[2] >= 24) :
            dist.append(inputparams[0] + inputparams[1] + inputparams[2] - 23.5)
            
        if inputparams[4] + inputparams[5] >= 24:
            dist.append(inputparams[4] + inputparams[5] - 23.5)
            
        if inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] >= 24:
            dist.append(inputparams[1] + inputparams[2] + inputparams[3] + inputparams[5] - 23.5)  
        
        distance = sum(dist)
        cost = arabid2lp_gates(inputparams) + distance
            
    return cost     #inputparams = [12.0,6.1,3.5,0.4,0.6,0.04,0.7]


gate = gatesm[152]

x_t = []
y_t = []
init_sol = []
domain = []
for j in range(1500):
    for i in range(1,7):
        globals()['x_%s' % i]  = random.uniform(0,24)
    for i in range(7,10):
        globals()['x_%s' % i]  = random.uniform(0,12)    
    for i in range(10,14):
        globals()['x_%s' % i]  = random.uniform(0,1)    
    for i in range(14,16):
        globals()['x_%s' % i]  = random.uniform(0,4)  
    while any([ x_1+ x_2 +x_3 >= 24, x_5 + x_6 >= 24, x_2 + x_3 + x_4 + x_6 >= 24 ]):
        for i in range(1,7):
            globals()['x_%s' % i]  = random.uniform(0,24)
        for i in range(7,10):
            globals()['x_%s' % i]  = random.uniform(0,12)    
        for i in range(10,14):
            globals()['x_%s' % i]  = random.uniform(0,1)    
        for i in range(14,16):
            globals()['x_%s' % i]  = random.uniform(0,4)  
    
    init_sol.append([globals()['x_%s' % i] for i in range(1,16) ])
    domain.append(arabid2lp([globals()['x_%s' % i] for i in range(1,16) ]))
init_sol = np.array(init_sol)    
domain = np.array(domain)
init_n_sol =  (init_sol - init_sol.min(0)) / init_sol.ptp(0)  
domain_n =  (domain - domain.min(0)) / domain.ptp(0)     
x_t = torch.Tensor(np.array(init_n_sol))
y_t = torch.Tensor(np.array(domain_n))

with open(f"Desktop/x_t_arabid2lp.txt", "wb") as fp:   #Pickling
    pickle.dump(x_t , fp)   
with open(f"Desktop/x_t_arabid2lp.txt", "rb") as fp:   # Unpickling
    x_t = pickle.load(fp)


x_train= x_t[0:1200]
y_train= y_t[0:1200]
x_test= x_t[1200:1500]
y_test = y_t[1200:1500]
    
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)



# Find optimal model hyperparameters
model.train()
likelihood.train()


# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 300

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
    observed_pred = likelihood(model(x_test))
    err = observed_pred.stddev
    f, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=200)
    ax.axline([0, 0], [1, 1],c = 'r')
    plt.errorbar(observed_pred.mean.numpy(), y_test.numpy(), yerr=err, fmt='.k',ecolor='blue', elinewidth=1.2,markersize='4') 
    ax.set_xlabel('Test data')
    ax.set_ylabel('Predicted data')
    ax.set_xticks(np.arange(0.0, 1.1, 0.1), minor=False)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1), minor=False)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.title(label = 'RBF Kernel')
    
    # Display graph
    
    plt.show()    




