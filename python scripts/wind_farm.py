#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:01:58 2021

@author: melikedila
"""

import numpy as np    # import required packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import math
from scipy.stats import multivariate_normal
import statistics as st
import pickle
import pymc3 as pm
#%matplotlib inline
from math import pi
from windrose import WindroseAxes
import theano.tensor as tt
import scipy.stats as st
from sklearn import mixture
from sklearn.mixture import GaussianMixture

df = pd.read_csv("/Users/melikedila/Desktop/la-haute-borne-data.csv")

wd = list(df.Wa_avg)  #  Absolute_wind_direction_corrected
ws = list(df.Ws_avg)   # Average wind speed

wd = [x for x in wd if math.isnan(x) == False]   
ws= [x for x in ws if math.isnan(x) == False]




data = list(zip(wd,ws))
dat =[]
for i in data:
    dat.append(list(i))
    
    
plt.plot(wd, ws, 'bx')
plt.axis('equal')
plt.show()

gmm = GaussianMixture(n_components=4)
gmm.fit(dat)


print(gmm.means_)
print('\n')
print(gmm.covariances_)



from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]);
plt.scatter(wd, ws);





from scipy.stats import pearsonr

P = pearsonr(wd, ws)

x = wd
y = ws

deltaX = (max(x) - min(x))/10
deltaY = (max(y)- min(y))/10

xmin = min(x) - deltaX 
xmax = max(x) + deltaX 

ymin = min(y) - deltaY
ymax = max(y) + deltaY


#create meshgrid

xx,yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]


positions= np.vstack([xx.ravel(),yy.ravel()])
values = np.vstack([x,y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T,xx.shape)


fig = plt.figure(dpi=100)
fig.set_figheight(1000)
ax = fig.gca()
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
cfset = ax.contourf(xx,yy,f,cmap='coolwarm')
ax.imshow(np.rot90(f),cmap = 'coolwarm',extent = [xmin,xmax,ymin,ymax])
cset = ax.contour(xx,yy, f, colors='k')
ax.clabel(cset,inline=1,fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.title('2D Gaussian Kernel density estimation')






























