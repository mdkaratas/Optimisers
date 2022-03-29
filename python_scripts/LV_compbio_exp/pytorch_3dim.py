#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:03:11 2022

@author: melikedila
"""

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
    t = 0.391
    #u = np.random.uniform(0,1)
    u = 0.621
    if neuro1lp_gates([x,y,z,t,u],[0,1])<4:
        init_sol.append([x,y,z]) 
        domain.append(neuro1lp_gates([x,y,z,t,u],[0,1]))
        c = c +1
init_sol = np.array(init_sol)    
# df = pd.DataFrame(init_sol)
# result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
domain = np.array(domain)#.reshape(-1, 1)
#init_n_sol = [float(i)/max(init_sol) for i in init_sol]#(init_sol - init_sol.min(0)) / init_sol.ptp(0)  #(init_sol - np.mean(init_sol, axis=0)) / np.std(init_sol, axis=0)  # StandardScaler().fit_transform(init_sol) #(init_sol - np.mean(init_sol, axis=0))#StandardScaler().fit_transform(init_sol)#(init_sol - np.mean(init_sol, axis=0)) / np.std(init_sol, axis=0) # (init_sol - init_sol.min(0)) / init_sol.ptp(0)  
init_n_sol = (init_sol - init_sol.min(0)) / init_sol.ptp(0)
#init_n_sol[:,3] = init_sol[:,3]
domain_n = StandardScaler().fit_transform(domain.reshape(-1, 1))#(domain
x_t = torch.Tensor(np.array(init_sol))
y_t = torch.Tensor(np.array(domain)).reshape(-1)

### save
import pickle
# with open(f"Desktop/x_t_3dim.txt", "wb") as fp:   
#   pickle.dump(x_t, fp)  
with open(f"Desktop/x_t_3dim.txt", "rb") as fp:   
 x_t = pickle.load(fp)   

# with open(f"Desktop/y_t_3dim.txt", "wb") as fp:   
#  pickle.dump(y_t, fp)  
with open(f"Desktop/y_t_3dim.txt", "rb") as fp:   
 y_t = pickle.load(fp)   


x_train= x_t[0:1700]
y_train= y_t[0:1700]
x_test= x_t[1700:2000]
y_test = y_t[1700:2000]



####    Class representing the model

class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # self.hid1 = torch.nn.Linear(3, 3)  # 3-(3-3)-1   # hidden layer
    # # self.drop1 = T.nn.Dropout(0.50)   ###  first dropout layer will ignore 0.50 (half) of randomly selected nodes in the hid1 layer on each call to forward() during training. 
    # self.hid2 = torch.nn.Linear(3, 3)   # hidden layer
    # self.oupt = torch.nn.Linear(3, 1)   # output layer
    self.layers = nn.Sequential(nn.Linear(3, 10),
                                nn.LeakyReLU(), #nn.GELU(),#nn.LeakyReLU(),#nn.LeakyReLU(),
                    nn.Linear(10, 10),
                    nn.LeakyReLU(),#nn.GELU(),#nn.LeakyReLU()
                   # nn.Linear(10, 10),
                   # nn.LeakyReLU(),
                   nn.Linear(10, 1))

    # torch.nn.init.xavier_uniform_(self.hid1.weight)   #  T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)
    # torch.nn.init.zeros_(self.hid1.bias)
    # torch.nn.init.xavier_uniform_(self.hid2.weight)
    # torch.nn.init.zeros_(self.hid2.bias)
    # torch.nn.init.xavier_uniform_(self.oupt.weight)
    # torch.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):     #  method that accepts input tensor(s) and computes output tensor(s)
    # z = torch.relu(self.hid1(x))
    # ###  z = self.drop1(z)
    # z = torch.relu(self.hid2(z))
    # #z = torch.sigmoid(z)  # no activation  -- In most neural regression problems, you don't apply an activation function to the output node---- have been normalized to a range of [0.0, 1.0], such as the demo data, it'd be feasible to apply sigmoid() activation
    # z = torch.(self.hid2(z))
    z = self.layers(x)
    return z



inp = x_t    # Create tensor of type torch.float32 
print('\nInput format: ', inp.shape, inp.dtype)     # Input format: torch.Size([150, 4]) torch.float32 
output = y_t    # Create tensor type torch.int64  
print('Output format: ', output.shape, output.dtype)  # Output format: torch.Size([150]) torch.int64 
data = TensorDataset(inp, output)    # Crea

# Split to Train, Validate and Test sets using random_split 
train_batch_size = 256        
number_rows = len(inp)    # The size of our dataset or the number of rows in excel table.  
test_split = int(number_rows*0.3)  
# validate_split = int(number_rows*0.2) 
train_split = number_rows - test_split    
train_set, test_set = random_split( data, [train_split, test_split])    

# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
# validate_loader = DataLoader(validate_set, batch_size = 1) 
test_loader = DataLoader(test_set, batch_size = 512)



###   Initializing the model, loss function and optimizer
# Initialize the MLP
mlp = MLP()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mlp.to(device)
# Define the loss function and optimizer
loss_function = nn.SmoothL1Loss() #nn.L1Loss() #nn.MSELoss()#nn.L2Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-6)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# optimizer =torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)  # adaptive gradient
# optimizer= torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)


#Function to save the model 
def saveModel(): 
    path = "Desktop/NetModel.pth" 
    torch.save(model.state_dict(), path) 





# Training Function 
def train(num_epochs): 
    best_accuracy = 0.0 
     
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_test_loss = 0.0 
        total = 0 
        #scheduler.step()
        # Training Loop 
        for data in train_loader: 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)   # predict output from the model 
            train_loss = loss_function(predicted_outputs, outputs)   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss +=train_loss.item()  # track the loss value 
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader) 
 
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for data in test_loader: 
               inputs, outputs = data 
               predicted_outputs = model(inputs) 
               test_loss = loss_function(predicted_outputs, outputs) 
             
               # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1) 
               running_test_loss += test_loss.item()  
               total += outputs.size(0) 
               running_accuracy += (predicted == outputs).sum().item() 
 
        # Calculate validation loss value 
        test_loss_value = running_test_loss/len(test_loader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)     
 
        # Save the model if the accuracy is the best 
        if accuracy > best_accuracy: 
            saveModel() 
            best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Test Loss is: %.4f' %test_loss_value)

if __name__ == "__main__": 
    num_epochs = 30000
    train(num_epochs) 
    print('Finished Training\n') 
    # test() 


# def test():     
#     # Load the model that we saved at the end of the training loop 
#     model = Network(input_size, output_size) 
#     path = "NetModel.pth" 
#     model.load_state_dict(torch.load(path)) 
     
#     running_accuracy = 0 
#     total = 0 
 
#     with torch.no_grad(): 
#         for data in test_loader: 
#             inputs, outputs = data 
#             outputs = outputs.to(torch.float32) 
#             predicted_outputs = model(inputs) 
#             _, predicted = torch.max(predicted_outputs, 1) 
#             total += outputs.size(0) 
#             running_accuracy += (predicted == outputs).sum().item() 
 
#         print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))    
 



path = "Desktop/NetModel.pth" 
torch.save(model.state_dict(), path) 

func = MLP()
func.load_state_dict(torch.load(path))
# #func.eval()

# func(x_t[0])


###  new bunch of data points

c = 0
while c<500:
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
    if neuro1lp_gates([x,y,z,t,u],[0,1])<4:
        init_sol.append([x,y,z]) 
        domain.append(neuro1lp_gates([x,y,z,t,u],[0,1]))
        c = c +1
init_sol = np.array(init_sol)    
# df = pd.DataFrame(init_sol)
# result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
domain = np.array(domain)#.reshape(-1, 1)