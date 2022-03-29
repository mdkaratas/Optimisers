#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:39:57 2022

@author: melikedila
"""

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset 

#%%
#######################  Data for the neuro1lp model
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
eng = matlab.engine.start_matlab()
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

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataDD.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingDD.mat')

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataDD']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingDD']

def neuro1lp_gates(inputparams,gates):
    inputparams = list(inputparams)
    inputparams = matlab.double([inputparams])
    gates = matlab.double([gates])
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost

# create data from the function
init_sol = []
domain = []
c = 0
while c<1000:
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = random.uniform(0,1)
    u = random.uniform(0,1)
    if neuro1lp_gates([x,y,z,t,u],[0,1])<4:
        init_sol.append([x,y,z,t,u]) 
        domain.append(neuro1lp_gates([x,y,z,t,u],[0,1]))
        c = c +1
init_sol = np.array(init_sol)    
domain = np.array(domain)#.reshape(-1, 1)

x_pts = torch.from_numpy(init_sol)
y_pts = torch.from_numpy(domain)



#%%



class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(nn.Linear(5, 6),
                                nn.LeakyReLU(),
                   nn.Linear(6, 6),
                   nn.LeakyReLU(),
                   nn.Linear(6, 1))

  def forward(self, x):     #  method that accepts input tensor(s) and computes output tensor(s)
    z = self.layers(x)
    return z



#model = MLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##  hyperparameters

learning_rate = 0.001
batch_size = 64
num_epochs = 1

##  Data formatting

data = TensorDataset(x_pts, y_pts) 
number_rows = len(x_pts)    # The size of our dataset or the number of rows in excel table.  
test_split = int(number_rows*0.3)  
train_split = number_rows - test_split    


train_set, test_set = random_split(data, [train_split, test_split])    
 
# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True) 
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)


##  initialise network

model = MLP()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

##  Train Network

####  Epoch = Network hyas seen all the data 
for epoch in range(num_epochs):
    running_train_loss = 0
    running_test_loss = 0.0 
    for inputs,targets in train_loader:  # enumerate to see batch index
        # data = data.to(device=device)
        # targets = targets.to(device=device)
        #print(data.shape)
        
        
        #forward
        scores = model(inputs)
        loss = loss_func(scores,targets)
        
        #backward
 
        optimizer.zerograd()   
        loss.backward()
        
        optimizer.step()
        running_train_loss +=loss.item()  # track the loss value 

    # Calculate training loss value 
    train_loss_value = running_train_loss/len(train_loader) 
    
    with torch.no_grad(): 
        model.eval() 
        for data,targets in test_loader: 
           inputs, outputs = data 
           predicted_outputs = model(inputs) 
           test_loss = loss_func(predicted_outputs, outputs) 
         
           # The label with the highest value will be our prediction 
           _, predicted = torch.max(predicted_outputs, 1) 
           running_test_loss += test_loss.item()  

    # Calculate validation loss value 
    test_loss_value = running_test_loss/len(test_loader) 
    
    # Calculate validation loss value 
    test_loss_value = running_test_loss/len(test_loader)     
    print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Test Loss is: %.4f' %test_loss_value)
            
        
















