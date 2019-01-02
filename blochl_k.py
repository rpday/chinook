#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 18:49:34 2018

@author: ryanday
"""

def score(point,N):
    (i0,j0,k0) = (0,0,0)
    if type(N)==int:
        
        (n1,n2,n3) = (N,N,N)
    elif iterable(N):
        if len(N)==3:
            (n1,n2,n3)= (N[0],N[1],N[2])
        
        
    (i,j,k) = point
    score = 1 + (i-i0)*0.5 + (n1+1)*((j-j0)*0.5 + (n2+1)*(k-k0)*0.5)
    return score