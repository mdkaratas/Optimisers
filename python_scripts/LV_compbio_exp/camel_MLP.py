#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:11:20 2022

@author: melikedila
"""

####  On another func 

def camel_back(x):
    return ( (4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2)


n = 1
k = 1



init_sol = []
domain = []
#for i in range(2000):
c = 0
while c<500:
    start_time = time.time()
    x0_cam_0 = np.random.uniform(low=-1.9, high=1.9, size=(k, n))
    x0_cam_1 = np.random.uniform(low=-1.1, high=1.1, size=(k, n))

    init = np.array([x0_cam_0, x0_cam_1]).ravel()

    init_sol.append(init) 
    domain.append(camel_back(init))
    c = c +1
init_sol = np.array(init_sol,dtype = 'float32')    
domain = np.array(domain,dtype = 'float32')#.reshape(-1, 1)


train_batch_size = 10      
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


train_set= TensorDataset(x_train, y_train)  
test_set= TensorDataset(x_test, y_test) 

# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
test_loader = DataLoader(test_set,  batch_size = 32)


n_data,n_inputs = init_sol.shape
# model definition
# class MLP(torch.nn.Module):
#     # define model elements
#     def __init__(self, n_inputs):
#         super(MLP, self).__init__()        
#         self.layers = nn.Sequential(nn.Linear(2, 10),
#                                     nn.LeakyReLU(),
#                         nn.Linear(10, 10),
#                         nn.LeakyReLU(),                
#                         nn.Linear(10, 1))
#     # forward propagate input
#     def forward(self, x):     #  method that accepts input tensor(s) and computes output tensor(s)
#       z = self.layers(x)
#       return z

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)
  
model = MLP()
    
loss_function = nn.SmoothL1Loss()#nn.smoothL1Loss#MSELoss()#nn.L2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)




def train(num_epochs): 
     
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        #running_accuracy = 0.0 
        running_test_loss = 0.0 
        total = 0 
 
        # Training Loop 
        for data in train_loader:
            #print(data)
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            
            inputs, outputs = inputs.float(), outputs.float()
            outputs = outputs.reshape((outputs.shape[0], 1))
            
            
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)   # predict output from the model 
            train_loss = loss_function(predicted_outputs, outputs)   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            
            # print out weights and biases
            # for name, param in model.named_parameters():
            #     print(name, param.grad.abs().sum())
            
            
            
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
                #_, predicted = torch.max(predicted_outputs, 1) 
                running_test_loss += test_loss.item()  
                total += outputs.size(0) 
                #running_accuracy += (predicted == outputs).sum().item() 
 
        # Calculate validation loss value 
        test_loss_value = running_test_loss/len(test_loader) 
                
        
        # Save the model if the accuracy is the best 
        # if accuracy > best_accuracy: 
        #     saveModel() 
        #     best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Test Loss is: %.4f' %test_loss_value)
        
        
if __name__ == "__main__": 
    num_epochs = 4000
    train(num_epochs) 
    print('Finished Training\n')   


#####  fit on the test data
pred = model(x_test)
pred = pred.detach().numpy()
# x_train = x_train.numpy()
# y_train = y_train.numpy()
# x_test = x_test.numpy()
# y_test = y_test.numpy()

plt.plot(x_test,y_test,'r',label ='Sine')
plt.plot(x_test,pred,'b',label ='cosine')
plt.axis('equal')
  
plt.show()

###### here fit on the training data
pred = model(x_train)
pred = pred.detach().numpy()

plt.plot(y_train,pred,'ro')
plt.xlim([0, 1])
plt.ylim([0, 1])
#plt.plot(x_train,y_train,'ro')
#plt.plot(x_test,y_test,'bo')
plt.show()


################


#####################
path = "Desktop/ML.pth" 
torch.save(model.state_dict(), path) 

model.load_state_dict(torch.load(path))
###################









pred = func(x_train)
pred = pred.detach().numpy()

plt.plot(x_train,y_train,'ro')
plt.plot(x_train,pred,'bo')
plt.xlim([0, 1])
plt.ylim([0, 1])
#plt.plot(x_train,y_train,'ro')
#plt.plot(x_test,y_test,'bo')
plt.show()
