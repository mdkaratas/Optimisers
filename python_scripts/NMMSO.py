# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:37:26 2021

@author: mk633
"""
from pynmmso import Nmmso
import numpy as np 
import matplotlib.pyplot 

class sphere:
    @staticmethod
    def fitness(x):
        return ( -1*((4-2.1*x[0]**2 + x[0]**4 /3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 -4)*x[1]**2))
    
    @staticmethod
    def get_bounds():
        return([-3,-2],[3,2])
    
    
n = Nmmso(sphere)
evolutions = 1000    

my_result = n.run(evolutions)

## mesela burada bir degiisklik yaptim

for r in my_result:
    print("Mode at {} has value {}".format(r.location, r.value))




#################

