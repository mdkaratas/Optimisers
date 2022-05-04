#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:34:49 2022

@author: mkaratas
"""



" This snippet finds the matrix which has sampled threshold combinations in the first 4th columns for 
  " 2lp Arabidopsis model and max normalised joint entropy for data in the 5th column ""
 
# Importing required libraries    
 
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
from pyDOE import *

eng = matlab.engine.start_matlab()

# Define the path

root = "/Users/mkaratas/Desktop/GitHub/"
path= root + r"BDEtools/code"
eng.addpath(path,nargout= 0)
path= root + r"BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= root + r"BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= root + r"BDE-modelling/Cost_functions/arabid2lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"BDEtools/models"
eng.addpath(path,nargout= 0)

#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataLL = eng.load('dataLL.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingLL = eng.load('lightForcingLL.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
link image 0 hasnt been detected
dataLL = dataLL['dataLL']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingLL=lightForcingLL['lightForcingLL']

#  Find joint entropy

j_ent = np.asarray(eng.arabid2lp_entropy_samples(dataLD, dataLL))
max_ent_idx = np.argmax(j_ent[:,4])
max_ent = j_ent[max_ent_idx]

Thresholds_arabid2lp = j_ent[max_ent_idx,:4]




