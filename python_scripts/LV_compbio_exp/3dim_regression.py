#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:41:57 2022

@author: melikedila
"""
########################################################################  Data loading and preparation for 3dim neuro1lp model-- MLP
import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset 
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
import pandas as pd
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

init_sol = []
domain = []
#for i in range(2000):
c = 0
while c<6000:
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    #t = np.random.uniform(0,1) 
    t = 0.391
    #u = np.random.uniform(0,1)
    u = 0.621
    if neuro1lp_gates([x,y,z,0.391,0.621],[0,1])<=4:
        init_sol.append([x,y,z]) 
        domain.append(neuro1lp_gates([x,y,z,0.391,0.621],[0,1]))
        c = c +1
init_sol = np.array(init_sol,dtype = 'float32')    
domain = np.array(domain,dtype = 'float32')#.reshape(-1, 1)


###############################################  Data save
with open(f"Desktop/xt_3dim.txt", "wb") as fp:   
  pickle.dump(init_sol, fp)  
with open(f"Desktop/xt_3dim.txt", "rb") as fp:   
 init_sol = pickle.load(fp)   

with open(f"Desktop/yt_3dim.txt", "wb") as fp:   
  pickle.dump(domain, fp)  
with open(f"Desktop/yt_3dim.txt", "rb") as fp:   
 domain = pickle.load(fp)   

 
 ###############################################  Data load
with open(f"Desktop/xt_3dim.txt", "rb") as fp:   
 init_sol = pickle.load(fp)   
with open(f"Desktop/yt_3dim.txt", "rb") as fp:   
 domain = pickle.load(fp)  


train_batch_size = 32       
number_rows = len(init_sol)   
test_split = int(number_rows*0.3)  
train_split = number_rows - test_split

########## adam akilli normlaisaiton !

def Normalise(x_tr):
    col = np.zeros((len(x_tr),x_tr.shape[1]),dtype = 'float32')
    for i in range(x_tr.shape[1]): #dim
        for j in range(x_tr.shape[0]): #length
            col[j,i] = (x_tr[j,i] - np.min(x_tr[:,i]))/ (np.max(x_tr[:,i]) - np.min(x_tr[:,i]))
    return col       




#normalizing input dataframe
x_tr = init_sol[0:train_split]
x_tr = Normalise(x_tr)
x_train = torch.from_numpy(x_tr)
#x_train = torch.from_numpy(init_sol[0:train_split])


##############################################################
############################################normalizing target dataframe
y_tr = domain[0:train_split]
col_ytr = np.zeros(len(y_tr),dtype = 'float32')
for j in range(len(y_tr)): #length
        col_ytr[j] = (y_tr[j] - np.min(y_tr))/ (np.max(y_tr) - np.min(y_tr))
y_tr = col_ytr        
y_train = torch.from_numpy(y_tr)

############################################normalizing tests
x_te = init_sol[train_split:test_split+train_split]
x_te = Normalise(x_te)
x_test = torch.from_numpy(x_te)

y_te = domain[train_split:test_split+train_split]
col_yte = np.zeros(len(y_te),dtype = 'float32')
for j in range(len(y_te)): #length
    col_yte[j] = (y_te[j] - np.min(y_te))/ (np.max(y_te) - np.min(y_te))
y_te = col_yte       
y_test = torch.from_numpy(y_te)

############################################################################################################

#standardize y
#y_tr = domain[0:train_split]
#y = StandardScaler().fit_transform(y_tr.reshape(-1, 1))
y_train = torch.from_numpy(domain[0:train_split])
x_test = torch.from_numpy(init_sol[train_split:test_split+train_split])
y_test = torch.from_numpy(domain[train_split:test_split+train_split])


train_set= TensorDataset(x_train, y_train)  
test_set= TensorDataset(x_test, y_test) 

# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
test_loader = DataLoader(test_set,  batch_size = 32)
###################################################################################################################   MLP

from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
 
n_data,n_inputs = init_sol.shape

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()        
        self.layers = nn.Sequential(nn.Linear(3, 15),
                                    nn.Sigmoid(),
                        nn.Linear(15, 10),
                         nn.Sigmoid(),
                         nn.Linear(10, 1)).to(device)
    # forward propagate input
    def forward(self, x):     #  method that accepts input tensor(s) and computes output tensor(s)
      z = self.layers(x)
      return z
 

 
# train the model
def train_model(train_loader, model):
    # define the optimization
    criterion = nn.SmoothL1Loss()#MSELoss()
    optimizer = SGD(model.parameters(), lr=1e-6, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_loader):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
 
# evaluate the model
def evaluate_model(test_loader, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_loader):
        print((inputs, targets))
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    #row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data

print(len(train_loader.dataset), len(test_loader.dataset))
# define the network
model = MLP(n_inputs)
# train the model
train_model(train_loader, model)
# evaluate the model
mse = evaluate_model(test_loader, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))


# make a single prediction (expect class=1)
# row = [0.00632,18.00,2.310]  ## Tensor comment out in predict function
# yhat = predict(row, model)
# print('Predicted: %.3f' % yhat)
# neuro1lp_gates([0.00632,18.00,2.310,0.391,0.621],[0,1])

###  See how good the approximation of data is by the emulator
path = "Desktop/NetModel.pth" 
torch.save(model.state_dict(), path) 

func = MLP(n_inputs)
func.load_state_dict(torch.load(path))
#func.eval()

pred = func(x_test)
pred = pred.detach().numpy()
x_train = x_train.numpy()
y_train = y_train.numpy()
x_test = x_test.numpy()
y_test = y_test.numpy()

plt.plot(x_test,y_test,'ro')
plt.plot(x_test,pred,'b')
plt.show()




###### here fit on the training data
pred = func(x_train)
pred = pred.detach().numpy()

plt.plot(y_train,pred,'ro')
plt.xlim([0, 1])
plt.ylim([0, 1])
#plt.plot(x_train,y_train,'ro')
#plt.plot(x_test,y_test,'bo')
plt.show()







predicted = predict(x_test,model)


plt.plot(x_test[:,2],y_test,'ro')
plt.plot(x_test[:,2],predicted,'b')
plt.show()



plt.plot(y_test,predicted,'ro')
plt.xlim([0, 12])
plt.ylim([0, 4])
plt.show()


######  normalised x_tests
x_t = init_sol[train_split:test_split+train_split]
df = pd.DataFrame(x_t)
result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(3))
x_tr = result.to_numpy()
x_test_ = torch.from_numpy(x_tr)


predicted = predict(x_test,model)[0]
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,predicted,'b')
plt.show()

################  non normalised data training and test