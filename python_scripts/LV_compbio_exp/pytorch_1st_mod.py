#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:06:11 2022

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
    cost=eng.getBoolCost_neuro1lp(inputparams,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
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
while c<1000:
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    #t = np.random.uniform(0,1) 
    t = random.uniform(0,1)
    #u = np.random.uniform(0,1)
    u = random.uniform(0,1)
    if neuro1lp_gates([x,y,z,t,u],[0,1])<4:
        init_sol.append([x,y,z,t,u]) 
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
import pickle
# with open(f"Desktop/x_t_5dim.txt", "wb") as fp:   
#   pickle.dump(x_t, fp)  
with open(f"Desktop/x_t_5dim.txt", "rb") as fp:   
 x_t = pickle.load(fp)   

# with open(f"Desktop/y_t_5dim.txt", "wb") as fp:   
#   pickle.dump(y_t, fp)  
with open(f"Desktop/y_t_5dim.txt", "rb") as fp:   
 y_t = pickle.load(fp)   



x_train= x_t[0:700]
y_train= y_t[0:700]
x_test= x_t[700:1000]
y_test = y_t[700:1000]



####    Class representing the model

class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # self.hid1 = torch.nn.Linear(3, 3)  # 3-(3-3)-1   # hidden layer
    # # self.drop1 = T.nn.Dropout(0.50)   ###  first dropout layer will ignore 0.50 (half) of randomly selected nodes in the hid1 layer on each call to forward() during training. 
    # self.hid2 = torch.nn.Linear(3, 3)   # hidden layer
    # self.oupt = torch.nn.Linear(3, 1)   # output layer
    self.layers = nn.Sequential(nn.Linear(3, 6),
                                nn.LeakyReLU(),
                   nn.Linear(6, 3),
                   nn.LeakyReLU(),
                   nn.Linear(3, 1))

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
#data = TensorDataset(inp, output)    # Crea

train_batch_size = 32       
number_rows = len(init_sol)   
test_split = int(number_rows*0.3)  
train_split = number_rows - test_split


#normalizing input dataframe
x_tr = init_sol[0:train_split]
df = pd.DataFrame(x_tr)
result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(3))
x_tr = result.to_numpy()
x_train = torch.from_numpy(x_tr)
#x_train = torch.from_numpy(init_sol[0:train_split])


##############################################################
############################################normalizing target dataframe
y_tr = init_sol[0:train_split]
df = pd.DataFrame(y_tr)
result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(3))
y_tr = result.to_numpy()
y_train = torch.from_numpy(y_tr)

############################################normalizing tests
x_tr = init_sol[train_split:test_split+train_split]
df = pd.DataFrame(x_tr)
result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(3))
x_tr = result.to_numpy()
x_test = torch.from_numpy(x_tr)

y_tr = init_sol[train_split:test_split+train_split]
df = pd.DataFrame(y_tr)
result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(3))
y_tr = result.to_numpy()
y_test= torch.from_numpy(y_tr)



train_set= TensorDataset(x_train, y_train)  
test_set= TensorDataset(x_test, y_test) 

# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
test_loader = DataLoader(test_set,  batch_size = 32)
###   Initializing the model, loss function and optimizer
# Initialize the MLP
mlp = MLP()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mlp.to(device)
#net.train()  # s


# Function to save the model 
def saveModel(): 
    path = "Desktop/NetModel.pth" 
    torch.save(model.state_dict(), path) 

# Define the loss function and optimizer
loss_function = nn.SmoothL1Loss()#nn.smoothL1Loss#MSELoss()#nn.L2Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)




# Training Function 
def train(num_epochs): 
    best_accuracy = 0.0 
     
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        #running_accuracy = 0.0 
        running_test_loss = 0.0 
        total = 0 
 
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
 

if __name__ == "__main__": 
    num_epochs = 30000
    train(num_epochs) 
    print('Finished Training\n') 
    # test() 


path = "Desktop/NetModel.pth" 
torch.save(model.state_dict(), path) 

func = MLP()
func.load_state_dict(torch.load(path))
#func.eval()

func(x_t[0])

plt.title('loss')
plt.plot(train_losses), plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.show()

# plotting test/train accuracy
plt.title('accuracy')
plt.plot(train_accs), plt.plot(test_accs)
plt.legend(['train', 'test'])
plt.show()





f, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=200)
ax.axline([0, 0], [1, 1],c = 'r')
ax.set_xlabel('Test data')
ax.set_ylabel('Predicted data')
ax.set_xticks(np.arange(0.0, 1.1, 0.1), minor=False)
ax.set_yticks(np.arange(0.0, 1.1, 0.1), minor=False)
ax.set_xlim([0, 1.1])
ax.set_ylim([0, 1.1])
plt.title(label = 'MLP')

# Display graph

plt.show()    

func = MLP()
###  new bunch of data points
x_t = []
y_t = []
init_sol = []
domain = []
#for i in range(2000):
c = 0
while c<500:
    start_time = time.time()
    x = random.uniform(0,24)
    y = random.uniform(0,24)
    while x+y > 24 :
        x = random.uniform(0,24)
        y = random.uniform(0,24)
    z = np.random.uniform(0,12) 
    t = np.random.uniform(0,1) 
    #t = 0.391
    u = np.random.uniform(0,1)
    #u = 0.621
    if neuro1lp_gates([x,y,z,t,u],[0,1])<4:
        init_sol.append([x,y,z,t,u]) 
        domain.append(neuro1lp_gates([x,y,z,t,u],[0,1]))
        c = c +1
init_sol = np.array(init_sol)    
# df = pd.DataFrame(init_sol)
# result = df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
domain = np.array(domain)#.reshape(-1, 1)

y = domain
init_sol = torch.Tensor(np.array(init_sol))
y_pred = []
for i in range(len(x_train)):
    print(i)
    y_pred.append(func(x_train[i]))

predictions = []
for i in range(500):
    predictions.append(y_pred[i].detach().numpy())
    
    
    
    
pr = []
for i in range(len(y_pred)):
    pr.append(y_pred[i].detach().numpy())    

import matplotlib.pyplot as plt

#plt.plot(domain,predictions, 'ro')
plt.plot(y_pred,y_train, 'ro')
plt.xlabel("common X")
plt.ylabel("common Y")
plt.axis('equal')
plt.xlabel("Data")
plt.ylabel("Prediction")
plt.show()

from torchsummary import summary
summary(func,input_size=(5,6,6))

arr_xtrain = x_train.numpy()
arr_ytrain = y_train.numpy()
arr_xtest = x_test.numpy()
arr_ytest = y_test.numpy()


b = np.array([0,1,2,3])
c = torch.from_numpy(b)














#####   creation of the training loop!


bat_size = 10
train_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)

net = Net().to(device)
net.train()  # set mode

lrn_rate = 0.005
loss_func = T.nn.MSELoss()
optimizer = T.optim.Adam(net.parameters(),
  lr=lrn_rate)

for epoch in range(0, 500):
  # T.manual_seed(1 + epoch)  # recovery reproduce
  epoch_loss = 0.0  # sum avg loss per item

  for (batch_idx, batch) in enumerate(train_ldr):
X = batch[0]  # predictors shape [10,8]
Y = batch[1]  # targets shape [10,1] 

optimizer.zero_grad()
oupt = net(X)            # shape [10,1]

loss_val = loss_func(oupt, Y)  # avg loss in batch
epoch_loss += loss_val.item()  # a sum of averages
loss_val.backward()
optimizer.step()

  if epoch % 100 == 0:
print(" epoch = %4d   loss = %0.4f" % \
 (epoch, epoch_loss))
# TODO: save checkpoint

print("\nDone ")






































####   preparing the dataset
  # Run the training loop
for epoch in range(0, 5): # 5 epochs at maximum
  
  # Print epoch
  print(f'Starting epoch {epoch+1}')
  
  # Set current loss value
  current_loss = 0.0
  
  # Iterate over the DataLoader for training data
  for i, data in enumerate(trainloader, 0):
    
    # Get and prepare inputs
    inputs, targets = data
    inputs, targets = inputs.float(), targets.float()
    targets = targets.reshape((targets.shape[0], 1))
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Perform forward pass
    outputs = mlp(inputs)
    
    # Compute loss
    loss = loss_function(outputs, targets)
    
    # Perform backward pass
    loss.backward()
    
    # Perform optimization
    optimizer.step()
    
    # Print statistics
    current_loss += loss.item()
#################################################################################################################
device = torch.device("cpu")
net = MLP().to(device)
net.train()
# AC=no, sqft=2500, style=bungalow, school=lincoln
x = torch.tensor([[0.0395, 0.5463, 0.8643]],
      dtype=torch.float32).to(device)
with torch:
  y = net(x)

print("\ninput = ")
print(x)
print("output = ")
print("%0.8f" % y.item())

x = torch.tensor([[0.2053, 0.6642, 0.9872],
              [0.0620, 0.7826, 0.6094]],
      dtype=torch.float32).to(device)
with torch:
  y = net(x)
print("\ninput = ")
print(x)
torch.set_printoptions(precision=8)
print("output = ")
print(y)

print("\nEnd test ")
#################################################################################################################
    

#####     In situations where a neural network model tends to overfit, you can use a technique called dropout. 
####     Model overfitting is characterized by a situation where model accuracy on the training data is good,
###        but model accuracy on the test data is poor.

###   You can add a dropout layer after any hidden layer. For example, 
###   to add two dropout layers to the demo network, you could modify the __init__() method like so:

####   Using dropout introduces randomness into the training which tends to make the trained model more resilient to new, 
  #####    previously unseen inputs. Because dropout is intended to control model overfitting, in most situations 
  ######   you define a neural network without dropout, and then add dropout only if overfitting seems to be happening




####   a loss function is used to compare model predictions and true targets - essentially computing how poor the model performs.
















net = Net().to(device)
net.eval()
# AC=no, sqft=2500, style=bungalow, school=lincoln
x = torch.tensor([[-1, 0.2500, 0,1,0, 0,0,1]],
      dtype=T.float32).to(device)
with torch.no_grad():
  y = net(x)

print("\ninput = ")
print(x)
print("output = ")
print("%0.8f" % y.item())

x = torch.tensor([[-1, 0.2500, 0,1,0, 0,0,1],
              [+1, 0.1060, 1,0,0, 0,1,0]],
      dtype=torch.float32).to(device)
with torch.no_grad():
  y = net(x)
print("\ninput = ")
print(x)
torch.set_printoptions(precision=8)
print("output = ")
print(y)

print("\nEnd test ")






class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
        
        
        
        
        
batch_size = 100
num_epochs = 100

        