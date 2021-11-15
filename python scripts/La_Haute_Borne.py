#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:23:11 2021

@author: melikedila
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import multivariate_normal
import statistics as st
import pickle

#df=f

df = pd.read_csv("Desktop/la-haute-borne-data.csv")

wd = list(df.Wa_avg)  #  Absolute_wind_direction_corrected
ws = list(df.Ws_avg)   # Average wind speed



wd = [x for x in wd if math.isnan(x) == False]   # true ya da false diye output veriyor eger deger nan is true
#print(wa)
ws= [x for x in ws if math.isnan(x) == False]



data = [wd,ws]
print(data[0])


plt.hist(df.Wa_avg)
plt.hist(df.Ws_avg)


Wa_avg = np.array(df.Wa_avg)
Ws_avg = np.array(df.Ws_avg)
#data = Wa_avg,Ws_avg
#print(data)

sns.displot(data)                



X = np.stack((wd, ws), axis=0)
Y = np.stack((Wa_avg, Ws_avg), axis=0)

cov_mat = np.cov(X)
mean_wd = sum(wd) / len(wd)
mean_ws = sum(ws) / len(ws)

cov_val = sum((a - mean_wd) * (b - mean_ws) for (a,b) in zip(wd,ws)) / len(wd)
cov_val = [24.749]
cov_mat = np.cov(X)




var_matrix=np.array([wd,ws]).T
mean = np.mean(var_matrix,axis=0)
sigma = np.cov(var_matrix.T)
dist = multivariate_normal(mean, cov=sigma)

x, y = np.mgrid[1:500:.05, 0:10:.05]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

z = dist.pdf(pos)

plt.contourf(x,y,z)
plt.show()




cov_mat[0]


ab = np.eye(2)



