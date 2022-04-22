#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:10:23 2022

@author: mkaratas
"""
import numpy as np
import math
import random

#############################   COCO Benchmarks


## Sphere


def Sphere(self, chromosome):
	"""F1 Sphere model
	unimodal, symmetric, separable"""
	fitness = 0
	for i in range(len(chromosome)):
		fitness += chromosome[i]**2
	return fitness

## Ellipsoidal


## Rastrigin 

' Here chromosome is the vector representing the solution'

def Rastrigin(self, chromosome):
	"""F5 Rastrigin's function
	multimodal, symmetric, separable"""
	fitness = 10*len(chromosome)
	for i in range(len(chromosome)):
		fitness += chromosome[i]**2 - (10*math.cos(2*math.pi*chromosome[i]))
	return fitness

## B¨uche-Rastrigin Function

## Linear Slope

## Ackley

def Ackley(x, y):
 return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
                np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20


## Eggholder

def Eggholder(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    return c

## Dropwave 

def Dropwave(x1,x2):
    b=0.5*(x1*x1+x2*x2)+2
    a=-(1+np.cos(12*np.sqrt(x1*x1+x2*x2)))/b
    return a



## Cross in Tray

def Cross_in_tray(x1,x2):
    a=np.fabs(100-np.sqrt(x1*x1+x2*x2)/np.pi)
    b=np.fabs(np.sin(x1)*np.sin(x2)*np.exp(a))+1
    c=-0.0001*b**0.1
    return c


## Buckin 

def Buckin(x1,x2):
    ab=np.fabs(x2-0.01*x1*x1)
    a=100*np.sqrt(ab)+0.01*np.fabs(x1+10)
    return a


## Six hump camel

def Sixhump(x1,x2): 
   return 4*x1**2-2.1*x1**4+(x1**6)/3+x1*x2-4*x2**2+4*x2**4



## Bird

def Bird(x,y):
 return np.sin(x)*(np.exp(1-np.cos(y))**2)+np.cos(y)*(np.exp(1-np.sin(x))**2)+(x-y)**2

## Himmelblau

def Himmelblau(x,y):
       return (((x**2+y-11)**2) + (((x+y**2-7)**2)))
   
    
## Quartic   ??

def Quartic(X):
    z = random.randint(0,1)
    sum =0
    for i in range(len(X)):
        sum = sum + (i * X[i]**4 )  
    return sum + z  










