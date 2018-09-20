# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:54:44 2018

@author: rday
"""

import numpy as np
from sympy.physics.quantum.spin import Rotation

def fact(N):
    '''
    Recursive factorial function for non-negative numbers!
    '''
    if N<0:
        return 0
    if N==0:
        return 1
    else:
        return N*fact(N-1)

def s_lims(j,m,mp):
    smin = min(0,m-mp)
    smax = max(0,min(j+m,j-mp))
    return np.arange(smin,smax,1)


def small_D(j,mp,m,B):
    
    pref = np.sqrt(fact(j+mp)*fact(j-mp)*fact(j+m)*fact(j-m))
    cos = np.cos(B/2.)
    sin = np.sin(B/2.)
    s = s_lims(j,m,mp)
    s_sum = sum([(-1)**(mp-m+sp)/den(j,m,mp,sp)*cos**(2*j+m-mp-2*sp)*sin**(mp-m+2*sp) for sp in s])
    return pref*s_sum

def den(j,m,mp,sp):
    return (fact(j+m-sp)*fact(sp)*fact(mp-m+sp)*fact(j-mp-sp))

def big_D(j,mp,m,A,B,y):
    return np.exp(-1.0j*mp*A)*small_D(j,mp,m,B)*np.exp(-1.0j*m*y)


def testing(j,A,B,y):
    sym_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    my_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    
    for m in range(-j,j+1):
        for mp in range(-j,j+1):
            sym_D[m+j,mp+j] = Rotation.D(j,mp,m,y,B,A).doit()
            my_D[m+j,mp+j] = big_D(j,mp,m,A,B,y)
            
    print(np.around(sym_D,4))
    print(np.around(my_D,4))
    
if __name__ == "__main__":
    
    A,B,y = (-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random())
    j = 2
    testing(j,A,B,y)

    