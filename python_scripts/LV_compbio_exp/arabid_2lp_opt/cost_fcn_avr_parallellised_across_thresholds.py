#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:17:49 2022

@author: melikedila
"""

num_T = 4   # number of thresholds
samples_T = 30
Threshs = lhs(num_T, samples=samples_T)
#Threshs = list(Threshs)
#agg_x =[] 
#%%



def arabid2lp_matlab(inputp,gates,dataLD,dataDD,lightForcingLD,lightForcingDD):
    print(inputp)
    inputp = list(inputp)
    inputp = matlab.double([inputp])
    gates = gate
    gates = matlab.double([gates])
    cost=eng.getBoolCost_cts_arabid2lp(inputp,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    return cost


def arabid2lp_costf(inputparams):
    #s = time.time()
    agg_cost =np.empty([Threshs.shape[0]])
    #gates = matlab.double([list(gate)])
    gates = gate
    #agg_x.append(inputparams)
    l = np.repeat(np.expand_dims(inputparams[9:11],axis=0),Threshs.shape[0],axis=0)
    inp = np.repeat(np.expand_dims(inputparams[0:9],axis=0),Threshs.shape[0],axis=0)
    inputs = np.append(np.append(inp,Threshs,axis=1),l,axis=1)
    #for thresh in range(Threshs.shape[0]):
        #agg_cost[thresh]= eng.getBoolCost_cts_arabid2lp(matlab.double([list(inputs[thresh])]),gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
    
    #all_args = [(matlab.double([list(inputs[thresh])]), gates, dataLD,dataDD,lightForcingLD,lightForcingDD) for thresh in range(Threshs.shape[0])]
    all_args = [(inputs[thresh], gates, dataLD,dataDD,lightForcingLD,lightForcingDD) for thresh in range(Threshs.shape[0])]
    num_cores = min(multiprocessing.cpu_count(), 32)
    agg_cost = Parallel(n_jobs=num_cores, backend="threading")(delayed(arabid2lp_matlab)(*args) for args in all_args)
    avr_cost = np.mean(agg_cost)
    #print('time passed : ',s - time.time())
    return avr_cost

def arabid2lp(inputparams):
    for i in inputparams:
        if (inputparams[0] + inputparams[2] < 24) :
            if (inputparams[1] + inputparams[3] < 24):
                cost=arabid2lp_costf(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24
                cost = dist + arabid2lp_costf(inputparams)
        else:
            if (inputparams[1] + inputparams[3] < 24):
                dist = (inputparams[0] + inputparams[2] - 24)
                cost = dist + arabid2lp_costf(inputparams)
            else:
                dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                cost = dist + arabid2lp_costf(inputparams)
    return cost


#arabid2lp_matlab(inputs[0],gates,dataLD,dataDD,lightForcingLD,lightForcingDD)


#inputparams = init_sol[0]
#arabid2lp(inputparams)

#%%   Asagiya bak np arrayle olusturma--- listle olusturma





#######################                  this is np array
for i in range(1):
    s = time.time()
    def arabid2lp_costf(inputparams):   ## this cost func creates np.array with all thresholds attached
        s = time.time()
        agg_cost =np.empty([Threshs.shape[0]])
        gates = matlab.double([list(gate)])
        #agg_x.append(inputparams)
        l = np.repeat(np.expand_dims(inputparams[9:11],axis=0),Threshs.shape[0],axis=0)
        inp = np.repeat(np.expand_dims(inputparams[0:9],axis=0),Threshs.shape[0],axis=0)
        inputs = np.append(np.append(inp,Threshs,axis=1),l,axis=1)
        for thresh in range(Threshs.shape[0]):
            #print(thresh)
            agg_cost[thresh]= eng.getBoolCost_cts_arabid2lp(matlab.double([list(inputs[thresh])]),gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        #min_val = min(agg_cost)
        #min_index = agg_cost.index(min_val)   
        avr_cost = np.mean(agg_cost)
        #print('time passed : ',time.time() - s)
        return avr_cost
    def arabid2lp(inputparams):
        for i in inputparams:
            if (inputparams[0] + inputparams[2] < 24) :
                if (inputparams[1] + inputparams[3] < 24):
                    cost=arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24
                    cost = dist + arabid2lp_costf(inputparams)
            else:
                if (inputparams[1] + inputparams[3] < 24):
                    dist = (inputparams[0] + inputparams[2] - 24)
                    cost = dist + arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                    cost = dist + arabid2lp_costf(inputparams)
        return cost

        
    inputparams = init_sol[0]
    print(arabid2lp(inputparams))
    print('time passed : ',time.time() - s)    
    
#######################                  this is list
for i in range(1):
    s = time.time()
    def arabid2lp_costf(inputparams):    #### this cost fcn creates each time inputs by appending
        #print(inputparams[3])
        agg_cost =[]
        gates = gate
        gates = matlab.double([list(gates)])
        #agg_x.append(inputparams)
        l = inputparams[9:11]
        inputparams_ = inputparams[0:9]
        inputparams_ = list(inputparams_)
        for id in range(len(Threshs)):
            inputparams_.append(float(Threshs[id][0])) # add thresholds here 
            inputparams_.append(float(Threshs[id][1]))
            inputparams_.append(float(Threshs[id][2]))
            inputparams_.append(float(Threshs[id][3]))
            inputparams_.append(l[0])
            inputparams_.append(l[1])
            inp = np.array(inputparams_)
            inputparams_ = matlab.double([inputparams_])
            agg_cost.append(eng.getBoolCost_cts_arabid2lp(inputparams_,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1))
            inputparams_ = list(inp)
        #min_val = min(agg_cost)
        #min_index = agg_cost.index(min_val)   
        avr_cost = np.mean(agg_cost)
        return avr_cost

    def arabid2lp(inputparams):
        for i in inputparams:
            if (inputparams[0] + inputparams[2] < 24) :
                if (inputparams[1] + inputparams[3] < 24):
                    cost=arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24
                    cost = dist + arabid2lp_costf(inputparams)
            else:
                if (inputparams[1] + inputparams[3] < 24):
                    dist = (inputparams[0] + inputparams[2] - 24)
                    cost = dist + arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                    cost = dist + arabid2lp_costf(inputparams)
        return cost

        
    inputparams = init_sol[0]
    print(arabid2lp(inputparams))
    print('time passed : ',time.time() - s)    
    
###################################################################################################       Paralel np array  --- paralel olmayan np array

#######################                  this is parallel np array

for i in range(1):
    s = time.time()
    def arabid2lp_matlab(inputp,gates,dataLD,dataDD,lightForcingLD,lightForcingDD):
        #print(inputp)
        inputp = list(inputp)
        inputp = matlab.double([inputp])
        #gates = gate
        gates = matlab.double([gates])
        cost=eng.getBoolCost_cts_arabid2lp(inputp,gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        return cost
    
    
    def arabid2lp_costf(inputparams):
        #s = time.time()
        agg_cost =np.empty([Threshs.shape[0]])
        #gates = matlab.double([list(gate)])
        gates = gate
        #agg_x.append(inputparams)
        l = np.repeat(np.expand_dims(inputparams[9:11],axis=0),Threshs.shape[0],axis=0)
        inp = np.repeat(np.expand_dims(inputparams[0:9],axis=0),Threshs.shape[0],axis=0)
        inputs = np.append(np.append(inp,Threshs,axis=1),l,axis=1)
        #for thresh in range(Threshs.shape[0]):
            #agg_cost[thresh]= eng.getBoolCost_cts_arabid2lp(matlab.double([list(inputs[thresh])]),gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        
        #all_args = [(matlab.double([list(inputs[thresh])]), gates, dataLD,dataDD,lightForcingLD,lightForcingDD) for thresh in range(Threshs.shape[0])]
        all_args = [(inputs[thresh], gates, dataLD,dataDD,lightForcingLD,lightForcingDD) for thresh in range(Threshs.shape[0])]
        num_cores = min(multiprocessing.cpu_count(), 32)
        agg_cost = Parallel(n_jobs=num_cores, backend="threading")(delayed(arabid2lp_matlab)(*args) for args in all_args)
        avr_cost = np.mean(agg_cost)
        #print('time passed : ',s - time.time())
        return avr_cost
    
    def arabid2lp(inputparams):
        for i in inputparams:
            if (inputparams[0] + inputparams[2] < 24) :
                if (inputparams[1] + inputparams[3] < 24):
                    cost=arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24
                    cost = dist + arabid2lp_costf(inputparams)
            else:
                if (inputparams[1] + inputparams[3] < 24):
                    dist = (inputparams[0] + inputparams[2] - 24)
                    cost = dist + arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                    cost = dist + arabid2lp_costf(inputparams)
        return cost    
    
    inputparams = init_sol[0]
    print(arabid2lp(inputparams))
    print('time passed : ',time.time() - s)    
    
    
    
    

#######################                  this is non-paralell np array
for i in range(1):
    s = time.time()
    def arabid2lp_costf(inputparams):   ## this cost func creates np.array with all thresholds attached
        s = time.time()
        agg_cost =np.empty([Threshs.shape[0]])
        gates = matlab.double([list(gate)])
        #agg_x.append(inputparams)
        l = np.repeat(np.expand_dims(inputparams[9:11],axis=0),Threshs.shape[0],axis=0)
        inp = np.repeat(np.expand_dims(inputparams[0:9],axis=0),Threshs.shape[0],axis=0)
        inputs = np.append(np.append(inp,Threshs,axis=1),l,axis=1)
        for thresh in range(Threshs.shape[0]):
            #print(thresh)
            agg_cost[thresh]= eng.getBoolCost_cts_arabid2lp(matlab.double([list(inputs[thresh])]),gates,dataLD,dataDD,lightForcingLD,lightForcingDD,nargout=1)
        #min_val = min(agg_cost)
        #min_index = agg_cost.index(min_val)   
        avr_cost = np.mean(agg_cost)
        #print('time passed : ',time.time() - s)
        return avr_cost
    def arabid2lp(inputparams):
        for i in inputparams:
            if (inputparams[0] + inputparams[2] < 24) :
                if (inputparams[1] + inputparams[3] < 24):
                    cost=arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24
                    cost = dist + arabid2lp_costf(inputparams)
            else:
                if (inputparams[1] + inputparams[3] < 24):
                    dist = (inputparams[0] + inputparams[2] - 24)
                    cost = dist + arabid2lp_costf(inputparams)
                else:
                    dist = inputparams[1] + inputparams[3] - 24 + inputparams[0] + inputparams[2] - 24
                    cost = dist + arabid2lp_costf(inputparams)
        return cost

        
    inputparams = init_sol[0]
    print(arabid2lp(inputparams))
    print('time passed : ',time.time() - s)       
    
    
    
    