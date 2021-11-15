# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:27:58 2020

@author: mk633
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## plot_land plots the landscape when intervals and function itself is given
def plot_land(fun,x_interval,y_interval):
    x_points = np.linspace(x_interval[0], x_interval[1], 100)
    y_points = np.linspace(y_interval[0], y_interval[1], 100)
    X, Y = np.meshgrid(x_points, y_points)
    Z = fun(X, Y)
    
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='jet_r', edgecolor=None)  # 'gist_earth'
    
    ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", 
           title="fitness value")

    return fig, ax
  
def ackley(x,y):
    return -20* np.exp(-0.2 * np.sqrt( 0.5 * (x**2+ y**2) )) - \
            np.exp( 0.5 * ( np.cos(2*np.pi*x)+ np.cos(2*np.pi*y))) + 20 + np.e
x_interval = (-1,1)
y_interval = (-1,1)
plot_land(ackley,x_interval,y_interval) 

def rastrigin(x,y):  # rast.m
    x = np.asarray_chkfinite(x)
    return 10*2 + ( (x**2 - 10 * np.cos( 2 * np.pi * x ))+ (y**2 - 10 * np.cos( 2 * np.pi * y )))       

x_interval = (1.5, 1.58)
y_interval = (0,0.1)

#plt.point.plot(x='x', y='y', ax=ax, style='r-')
plot_land(rastrigin,x_interval,y_interval) 
#plt.pyplot(1.57174,0.00575191)
def camel_back(x,y):
    return ( (4-2.1*x**2 + x**4 /3)*x**2 + x*y + (4*y**2 -4)*y**2)
x_interval = (-2,-1)
y_interval = (-0.5, 0.5)
plot_land(camel_back,x_interval,y_interval) 
plt.contour(1.57174,0.00575191, 0.13291113660065335,)
####'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
           # 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
           # 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'

x, y = numpy.mgrid[-2:-1:20*1j, -4:4:20*1j]

# Draw the scalar field level curves
cs = plt.contour(scalar_field, extent=[-4, 4, -4, 4])
plt.clabel(cs, inline=1, fontsize=10)
      
#####################################################################


#plt.plot(105,200,'ro') 

  

