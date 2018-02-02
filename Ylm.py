#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:46:21 2017

@author: ryanday
"""

import numpy as np
import scipy.special as sc
from math import factorial

def Y(l,m,theta,phi):
    if l == 0:
        if m==0:
            tmp = 0.5*np.sqrt(1.0/np.pi)
        else:
            tmp = 0.0    
    elif l == 1:
        if m == -1:
            tmp = 0.5*np.sqrt(3.0/(2*np.pi))*np.exp(-1.0j*phi)*np.sin(theta)
        elif m == 0:
            tmp = 0.5*np.sqrt(3.0/np.pi)*np.cos(theta)
        elif m == 1:
            tmp = -0.5*np.sqrt(3.0/(2*np.pi))*np.exp(1.0j*phi)*np.sin(theta)
        else:
            tmp = 0.0
    elif l == 2:
        if m == -2:
            tmp = 0.25*np.sqrt(15.0/(2*np.pi))*np.exp(-2.0j*phi)*np.sin(theta)**2
        elif m == -1:
            tmp = 0.5*np.sqrt(15.0/(2*np.pi))*np.exp(-1.0j*phi)*np.sin(theta)*np.cos(theta)
        elif m==0:
            tmp = 0.25*np.sqrt(5/np.pi)*(3*np.cos(theta)**2-1)
        elif m == 1:
            tmp = -0.5*np.sqrt(15.0/(2*np.pi))*np.exp(1.0j*phi)*np.sin(theta)*np.cos(theta)
        elif m == 2:
            tmp = 0.25*np.sqrt(15.0/(2*np.pi))*np.exp(2.0j*phi)*np.sin(theta)**2
        else:
            tmp = 0.0
    elif l == 3:
        if m == -3:
            tmp = 1.0/8*np.sqrt(35/np.pi)*np.exp(-3.0j*phi)*np.sin(theta)**3
        elif m == -2:
            tmp = 1.0/4*np.sqrt(105/(2*np.pi))*np.exp(-2.0j*phi)*np.sin(theta)**2*np.cos(theta)
        elif m == -1:
            tmp = 1.0/8*np.sqrt(21/np.pi)*np.exp(-1.0j*phi)*np.sin(theta)*(5*np.cos(theta)**2-1)
        elif m == 0:
            tmp = 1.0/4*np.sqrt(7/np.pi)*(5*np.cos(theta)**3-3*np.cos(theta))
        elif m == 1:
            tmp = -1.0/8*np.sqrt(21/np.pi)*np.exp(1.0j*phi)*np.sin(theta)*(5*np.cos(theta)**2-1)
        elif m == 2:
            tmp = 1.0/4*np.sqrt(105/(2*np.pi))*np.exp(2.0j*phi)*np.sin(theta)**2*np.cos(theta) 
        elif m == 3:
            tmp = -1.0/8*np.sqrt(35/np.pi)*np.exp(3.0j*phi)*np.sin(theta)**3
        else:
            tmp = 0.0
    else:
        tmp = 0.0
    return tmp


def j(n,x):
    if n==0:
        tmp = np.sinc(x)
    elif n==1:
        tmp = -np.cos(x)/x+np.sinc(x)/x
    elif n==2:
        tmp = (3./x**2-1)*np.sinc(x)-3/x**2*np.cos(x)
    elif n==3:
        tmp = (15./x**3-6./x)*np.sinc(x)-(15/x**2-1)*np.cos(x)/x
    return tmp

def binom(a,b):
    return factorial(a+b)/float(factorial(a-b)*factorial(b))

def laguerre(x,l,j):
#    tmp = [binom(l+j,j-i) for i in range(j+1)]
    tmp = sum([((-1)**i)*(binom(l+j,j-i)*x**i/float(factorial(i))) for i in range(j+1)])
    return tmp

if __name__=="__main__":
    x = np.linspace(0,5,100)
    tmp = laguerre(x,5,0)
#    th = np.random.random()*np.pi
#    ph = np.random.random()*2*np.pi
#    for i in range(4):
#        for j in range(-i,i+1):
#            Yme = Y(i,j,th,ph) 
#            Ysc = sc.sph_harm(j,i,ph,th)
#            diff = abs(Yme-Ysc)
#            print i,j,diff
#    