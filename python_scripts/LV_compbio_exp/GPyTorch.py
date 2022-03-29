#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 23:06:44 2022

@author: melikedila
"""

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
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
import pandas as pd
eng = matlab.engine.start_matlab()


#%%  

# Set required paths

path= r"/Users/melikedila/Documents/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/GitHub/BDE-modelling/Cost_functions/neuro1lp_costfcn"
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

########
def neuro1lp_gates(inputparams,gates):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost


numsamps = 1000
samplemat = eng.neuro1lp_entropy_samples(dataLD, dataDD, numsamps,nargout=1)
sample_mat = []
for i in samplemat:
    sample_mat.append(i)
#  [0.39168151741939117,0.6213507923956284,0.9988925017099197]

x_t = []
y_t = []
init_sol = []
domain = []
#for i in range(2000):
c = 0
while c<2000:
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    #t = np.random.uniform(0,1) 
    t = 0.3916
    #u = np.random.uniform(0,1)
    u = 0.6213
    if neuro1lp_gates([x,y,z,0.3916,0.6213],[0,1])<4:
        init_sol.append([x,y,z]) 
        domain.append(neuro1lp_gates([x,y,z,0.3916,0.6213],[0,1]))
        c = c +1
init_sol = np.array(init_sol)    
# df = pd.DataFrame(init_sol)
# result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
domain = np.array(domain)#.reshape(-1, 1)
#init_n_sol = [float(i)/max(init_sol) for i in init_sol]#(init_sol - init_sol.min(0)) / init_sol.ptp(0)  #(init_sol - np.mean(init_sol, axis=0)) / np.std(init_sol, axis=0)  # StandardScaler().fit_transform(init_sol) #(init_sol - np.mean(init_sol, axis=0))#StandardScaler().fit_transform(init_sol)#(init_sol - np.mean(init_sol, axis=0)) / np.std(init_sol, axis=0) # (init_sol - init_sol.min(0)) / init_sol.ptp(0)  
init_n_sol = (init_sol - init_sol.min(0)) / init_sol.ptp(0)
#init_n_sol[:,3] = init_sol[:,3]
domain_n = StandardScaler().fit_transform(domain.reshape(-1, 1))#(domain - domain.min(0)) / domain.ptp(0) #StandardScaler().fit_transform(domain.reshape(-1, 1)) #(domain - domain.min(0)) / domain.ptp(0) #StandardScaler().fit_transform(domain) #(domain - domain.min(0)) / domain.ptp(0)     
x_t = torch.Tensor(np.array(init_n_sol))
y_t = torch.Tensor(np.array(domain_n)).reshape(-1)

x_train= x_t[0:1700]
y_train= y_t[0:1700]
x_test= x_t[1700:2000]
y_test = y_t[1700:2000]


with open("Desktop/x_vals.txt", "wb") as fp:   
    pickle.dump(x_t, fp)  
    
    
with open("Desktop/y_vals.txt", "wb") as fp:   
    pickle.dump(y_t, fp)     


import seaborn as sns
sns.scatterplot(x=init_sol[:,0], y= init_sol[:,1]);


# # normalize features
# mean = x_train.mean(dim=-2, keepdim=True)
# std = y_train.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std


# # normalize labels
# mean, std = train_y.mean(),train_y.std()
# train_y = (train_y - mean) / std
# test_y = (test_y - mean) / std

# # make continguous
# train_x, train_y = train_x.contiguous(), train_y.contiguous()
# test_x, test_y = test_x.contiguous(), test_y.contiguous()


##########################

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

training_iter = 600

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    #loss.backward()
    loss.mean().backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.mean().item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()   ## this is to initiate gradient descent 



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()


##########################
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(x_test))
#     err = observed_pred.stddev
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
#     plt.title(label = 'RBF')
#     plt.show()

##################
with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
    
    observed_pred = likelihood(model(x_test))
    err = observed_pred.stddev
    f, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=200)
    ax.axline([0, 0], [1, 1],c = 'r')
    plt.errorbar(observed_pred.mean.numpy(), y_test.numpy(), yerr=err, fmt='.k',ecolor='blue', elinewidth=1.2,markersize='4') 
    ax.set_xlabel('Data')
    ax.set_ylabel('Prediction')
    ax.set_xticks(np.arange(-4, 3, 0.8), minor=False)
    ax.set_yticks(np.arange(-4, 3, 0.8), minor=False)
    ax.set_xlim([-4, 3])
    ax.set_ylim([-4, 3])
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
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

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
    print(model(x_test))
    observed_pred = likelihood(model(x_test))
    err = observed_pred.stddev
    f, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=200)
    ax.axline([0, 0], [1, 1],c = 'r')
    plt.errorbar(observed_pred.mean.numpy(), y_test.numpy(), yerr=err, fmt='.k',ecolor='blue', elinewidth=1.2,markersize='4') 
    ax.set_xlabel('Test data')
    ax.set_ylabel('Predicted data')
    ax.set_xticks(np.arange(-4, 2, 0.8), minor=False)
    ax.set_yticks(np.arange(-4, 2, 0.8), minor=False)
    ax.set_xlim([-4,2])
    ax.set_ylim([-4,2])
    plt.title(label = 'Matern Kernel')
    
    # Display graph
    
    plt.show()    

#############################################  Gate [1,0]

x_t = []
y_t = []
init_sol = []
domain = []
for i in range(1500):
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    u = np.random.uniform(0,1)
    init_sol.append([x,y,z,t,u]) 
    domain.append(neuro1lp_gates([x,y,z,t,u],[1,0]))
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




#########################

# imports for training
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

from pytorch_forecasting.data.examples import get_stallion_data
data = get_stallion_data()  # load data as pandas dataframe





































