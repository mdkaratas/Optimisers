#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:45:04 2022

@author: mkaratas
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
path= root + r"BDE-modelling/Cost_functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= root + r"BDE-modelling/Cost_functions/costfcn_routines"
eng.addpath(path,nargout= 0)
path= root + r"BDEtools/models"
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

#  Find joint entropy

j_ent = np.asarray(eng.neuro1lp_entropy_samples(dataLD, dataDD))
max_ent_idx = np.argmax(j_ent[:,2])
max_ent = j_ent[max_ent_idx]

Thresholds_neuro1lp = j_ent[max_ent_idx,:2]

