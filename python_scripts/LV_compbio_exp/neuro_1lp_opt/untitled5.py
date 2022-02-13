#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:46:17 2022

@author: melikedila
"""

import pickle

with open("Desktop/l_results/avr_dist_50_1.txt", "rb") as fp:   
    avr_dist_50_1 = pickle.load(fp)   
with open("Desktop/l_results/avr_dist_50_2.txt", "rb") as fp:   
    avr_dist_50_2 = pickle.load(fp)  
with open("Desktop/l_results/avr_dist_50_3.txt", "rb") as fp:   
    avr_dist_50_3 = pickle.load(fp)   
with open("Desktop/l_results/avr_dist_50_4.txt", "rb") as fp:   
    avr_dist_50_4 = pickle.load(fp)  
with open("Desktop/l_results/avr_dist_50_5.txt", "rb") as fp:   
    avr_dist_50_5 = pickle.load(fp) 
    
    
    
with open("Desktop/l_results/avr_dist_100_1.txt", "rb") as fp:   
    avr_dist_100_1 = pickle.load(fp)  
with open("Desktop/l_results/avr_dist_100_2.txt", "rb") as fp:   
    avr_dist_100_2 = pickle.load(fp)  
with open("Desktop/l_results/avr_dist_100_3.txt", "rb") as fp:   
    avr_dist_100_3 = pickle.load(fp)  
with open("Desktop/l_results/avr_dist_100_4.txt", "rb") as fp:   
    avr_dist_100_4 = pickle.load(fp)   
with open("Desktop/l_results/avr_dist_100_5.txt", "rb") as fp:   
    avr_dist_100_5 = pickle.load(fp)    
    
    
    
    
    
    
plt.plot([50,100,1000,5000,10000],[0.008,0.002,0,0,0],'ro')    
plt.xlabel('Sample size')
plt.ylabel('p')
plt.show()
