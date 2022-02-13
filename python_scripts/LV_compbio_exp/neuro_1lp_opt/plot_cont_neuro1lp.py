#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:18:55 2022

@author: melikedila
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
import seaborn as sns
import pandas as pd
import plotly.express as px
import pandas
from pandas.plotting import parallel_coordinates
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import statistics as st
import matlab
#%%
#####    Readings and the plots

###      NMMSO

# without penalty

#############################################  NMMSO MI without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO_MI_cont.txt", "rb") as fp:   
    design_dict_NMMSO_MI = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO_MI_cont.txt", "rb") as fp:   
    fit_dict_NMMSO_MI = pickle.load(fp)
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_MI_cont.txt", "rb") as fp:   
    x_NMMSO_MI = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_MI_cont.txt", "rb") as fp:   
    f_NMMSO_MI  = pickle.load(fp)         
    
#############################################  NMMSO 01 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO1_cont.txt", "rb") as fp:   
    design_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO1_cont.txt", "rb") as fp:   
    fit_dict_NMMSO1 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_1_cont.txt", "rb") as fp:   
    x_NMMSO_1 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_1_cont.txt", "rb") as fp:   
    f_NMMSO_1 = pickle.load(fp)  
    
      
#############################################  NMMSO 10 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO2_cont.txt", "rb") as fp:   
    design_dict_NMMSO2 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO2_cont.txt", "rb") as fp:   
    fit_dict_NMMSO2 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_2_cont.txt", "rb") as fp:   
    x_NMMSO_2 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_2_cont.txt", "rb") as fp:   
    f_NMMSO_2 = pickle.load(fp)  
    


#############################################  NMMSO 11 without penalty

with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO2_cont.txt", "rb") as fp:   
    design_dict_NMMSO3 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO2_cont.txt", "rb") as fp:   
    fit_dict_NMMSO3 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_2_cont.txt", "rb") as fp:   
    x_NMMSO_3 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_2_cont.txt", "rb") as fp:   
    f_NMMSO_3 = pickle.load(fp)  
    

#############################################  NMMSO 00 without penalty


with open("Desktop/Llyonesse/Neuro_1lp_res/cont/design_dict_NMMSO0_cont.txt", "rb") as fp:   
    design_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/fit_dict_NMMSO0_cont.txt", "rb") as fp:   
    fit_dict_NMMSO0 = pickle.load(fp)  
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/x_NMMSO_0_cont.txt", "rb") as fp:   
    x_NMMSO_0 = pickle.load(fp)      
with open("Desktop/Llyonesse/Neuro_1lp_res/cont/f_NMMSO_0_cont.txt", "rb") as fp:   
    f_NMMSO_0 = pickle.load(fp)  


x_NMMSO_1[18]
x_NMMSO_1[21]




















    