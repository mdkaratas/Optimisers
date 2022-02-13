#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melikedila
"""

import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()


path= r"/Users/melikedila/Documents/Repos/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/models"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions/example"
eng.addpath(path,nargout= 0)


#%%

dataDD = eng.load('dataDD.mat')
dataDD = dataDD ['dataDD']
dat0 = dataDD[0]
dat1 = dataDD[1]
dat2 = dataDD[2]
dataD = [dat0,dat1,dat2]
dataLD = eng.load('dataLD.mat')
dataLD = dataLD['dataLD']
dt0 = dataLD[0]
dt1 = dataLD[1]
dt2 = dataLD[2]
dataL = [dt0,dt1,dt2]
lightForcingDD = eng.load('lightForcingDD.mat')
lightForcingDD = lightForcingDD ['lightForcingDD']
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingLD = lightForcingLD ['lightForcingLD']


#%%

inputparams=[5.0752, 6.0211, 14.5586, 0.3, 0.3]
inputparams=matlab.double([inputparams])
gates_1lp=matlab.logical([0, 1])

lightForcingD= {}
lightForcingD['x'] = lightForcingDD['x'][0]
lightForcingD['y'] = lightForcingDD['y'][0]

lightForcingL = {}
lightForcingL['x'] = lightForcingLD['x'][0]
lightForcingL['y'] = lightForcingLD['y'][0]
#%%

#cost=eng.getBoolCost_neuro1lp(inputparams[0],gates[0],dataD,dataL,lightForcingD,lightForcingL)
#{"mode":"full","isActive":false}
cost=eng.getBoolCost_neuro1lp(inputparams[0],gates[0],dataDD,dataLD,lightForcingD,lightForcingL)
#cost=eng.getBoolCost_neuro1lp(inputparams,gates,nargout=1)
#costfcn_bool_incl_thresh(model, data, delays, thresholds, lightForcing, dataTimePoints, varargin)

delays=[5.0752, 6.0211, 14.5586]
delays = matlab.double([delays])

gates_2lp=matlab.logical([[0,0,1,1,1])

#                               FUNCTIONS

def calc_getBoolCost(model,gates,inputparams,delays):
    if model == 'neuro1lp' :
        cost = cost=eng.getBoolCost_neuro1lp(inputparams[0],gates[0],dataDD,dataLD,lightForcingD,lightForcingL)
    if model == 'neuro2lp' :
        cost = cost=eng.getBoolCost_neuro2lp(inputparams[0],gates[0],dataDD,dataLD,lightForcingD,lightForcingL)
    return cost
    

c = calc_getBoolCost('neuro2lp',inputparams,delays)














