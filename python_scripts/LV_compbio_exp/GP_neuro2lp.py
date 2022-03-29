#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:58:53 2022

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

#gate = sys.argv[1]

##  all gate combinations of length of n
n = 5
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
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro2lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/models"
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


gate = gatesm[6]
    
def neuro2lp_gates(inputparams):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = gate
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_neuro2lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

def neuro2lp(inputparams):
    for i in inputparams:
        if (inputparams[0] + inputparams[2] < 24) :
            if (inputparams[1] + inputparams[3] < 24):
                cost=neuro2lp_gates(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24
                cost = dist + neuro2lp_gates(inputparams)
        else:
            if (inputparams[1] + inputparams[3] < 24):
                dist = (inputparams[0] + inputparams[2] - 24)
                cost = dist + neuro2lp_gates(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                cost = dist + neuro2lp_gates(inputparams)
    return cost     #i
    
x_t = []
y_t = []
init_sol = []
domain = []
for i in range(1500):
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    z = random.uniform(0,24)
    while x+z > 24 :
        x = random.uniform(0,24)
        z = random.uniform(0,24)  
    t = random.uniform(0,24)
    while y+t > 24 :
        y = random.uniform(0,24)
        t = random.uniform(0,24)    
    u = np.random.uniform(0,12) 
    v = np.random.uniform(0,1) 
    p = np.random.uniform(0,1)
    w = np.random.uniform(0,1)
    init_sol.append([x,y,z,t,u,v,p,w]) 
    domain.append(neuro2lp([x,y,z,t,u,v,p,w]))
init_sol = np.array(init_sol)    
domain = np.array(domain)
init_n_sol =  (init_sol - init_sol.min(0)) / init_sol.ptp(0)  
domain_n =  (domain - domain.min(0)) / domain.ptp(0)     
x_t = torch.Tensor(np.array(init_n_sol))
y_t = torch.Tensor(np.array(domain_n))

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



###################################################   Matern kernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

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


# ##########################
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(x_test))
#     err = observed_pred.variance
#     f, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=200)
    
    
#     #lower, upper = observed_pred.confidence_region()
    
#     # Plot training data as black stars
#     ax.plot(observed_pred.mean.numpy(), y_test.numpy(), 'k*',label='RBF')
#     ax.set_xlabel('Test data')
#     ax.set_ylabel('Predicted data')
#     ax.set_xticks(np.arange(0.0, 1.1, 0.1), minor=False)
#     ax.set_yticks(np.arange(0.0, 1.1, 0.1), minor=False)
#     ax.axline([0, 0], [1, 1],c = 'r')
#     #ax.fill_between(y_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     #ax.legend(['Observed Data', 'Test Data', 'Confidence'])
#     f.tight_layout()
#     plt.title(label = 'Matern Kernel')
#     plt.errorbar(observed_pred.mean.numpy(), y_test.numpy(), yerr = err,fmt='o',ecolor = 'blue')
#     plt.show()
    
    
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
    plt.title(label = 'Matern Kernel')
    
    # Display graph
    
    plt.show()    




















    
    
    
    