#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:46:21 2017

@author: ryanday
MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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


def gaunt(l,m,dl,dm):
    '''
    I prefer to avoid using the sympy library where possible. These are the explicitly defined
    Gaunt coefficients required for dipole-allowed transitions (dl = +/-1) for arbitrary m,l and dm
    These have been tested against the sympy package to confirm numerical accuracy for all l,m possible
    up to l=5. This function is equivalent, for the subset of dm, dl allowed to
    sympy.physics.wigner.gaunt(l,1,l+dl,m,dm,-(m+dm))
    args:
        l: int orbital angular momentum quantum number
        m: int azimuthal angular momentum quantum number
        dl: int change in l (+/-1)
        dm: int change in azimuthal angular momentum (-1,0,1)
    return:
        float Gaunt coefficient
    '''
    try:
        if abs(m + dm)<=(l+dl):
            if dl==1:
                if dm == 1:
                    return (-1.)**(m+1)*np.sqrt(3*(l+m+2)*(l+m+1)/(8*np.pi*(2*l+3)*(2*l+1)))
                elif dm == 0:
                    return (-1.)**(m)*np.sqrt(3*(l-m+1)*(l+m+1)/(4*np.pi*(2*l+3)*(2*l+1)))
                elif dm == -1:
                    return (-1.)**(m-1)*np.sqrt(3*(l-m+2)*(l-m+1)/(8*np.pi*(2*l+3)*(2*l+1)))
            elif dl == -1:
                if dm==1:
                    return (-1.)**(m)*np.sqrt(3*(l-m)*(l-m-1)/(8*np.pi*(2*l+1)*(2*l-1)))
                elif dm == 0:
                    return (-1.)**(m)*np.sqrt(3*(l+m)*(l-m)/(4*np.pi*(2*l+1)*(2*l-1)))
                elif dm == -1:
                    return (-1.)**(m)*np.sqrt(3*(l+m)*(l+m-1)/(8*np.pi*(2*l+1)*(2*l-1)))
        else:
            return 0.0
    except ValueError:
        print('Invalid entries for dipole matrix element-related Gaunt coefficients')
        print('l = {:0.4f}, m = {:0.4f}, dl = {:0.4f}, dm = {:0.4f}'.format(l,m,dl,dm))
        return 0.0
    
    

    
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