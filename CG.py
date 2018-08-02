# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:43:36 2018

@author: rday
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
import datetime as dt
import sympy.physics.wigner as wig


def CG_table(l1,l2,l,m1,m2,m):
    if l2 == 0:
        if l==l1 and m==m1:
            return 1
    if l1==1 and l2==1:
        if m==2:
            if m1==1 and m2==1 and l==2:
                return 1
        if m==1:
            if m1==1 and m2==0 and (l==2 or l==1):
                return np.sqrt(0.5)
            if m1==0 and m1==1:
                if l2==2:
                    return np.sqrt(0.5)
                elif l2==1:
                    return -np.sqrt(0.5)
        if m==0:
            if m1==1 and m2==-1:
                if l==2:
                    return np.sqrt(1./6)
                elif l==1:
                    return np.sqrt(0.5)
                elif l==0:
                    return np.sqrt(1./3)
            elif m1==0 and m2==0:
                if l==2:
                    return np.sqrt(2./3)
                elif l==0:
                    return -np.sqrt(1./3)
            elif m1==-1 and m2==1:
                if l==2:
                    return np.sqrt(1./6)
                elif l==1:
                    return -np.sqrt(1./2)
                elif l==0:
                    return np.sqrt(1./3)
        if l1==2 and l2 == 1:
            if m == 3:
                if m1==2 and m1==1 and l==3:
                    return 1.
            elif m==2:
                if m1==2 and m2==0:
                    if j==3:
                        return np.sqrt(1./3)
                    elif j==2:
                        return np.sqrt(2./3)
                elif m1==1 and m2==1:
                    if j==3:
                        return np.sqrt(2./3)
                    elif j==2:
                        return -np.sqrt(1./3)
            elif m==1:
                if m1==2 and m2==-1:
                    if j==3:
                        return np.sqrt(1./15)
                    elif j==2:
                        return np.sqrt(1./3)
                    elif j==1:
                        return np.sqrt(3./5)
                elif m1==1 and m2==0:
                    if j==3:
                        return np.sqrt(8./15)
                    elif j==2:
                        return np.sqrt(1./6)
                    elif j==1:
                        return -np.sqrt(3./10)
                elif m1==0 and m2==1:
                    if j==3:
                        return np.sqrt(2./5)
                    elif j==2:
                        return -np.sqrt(0.5)
                    elif j==1:
                        return np.sqrst(1./10)
            elif m==0:
                if m1==1 and m2==-1:
                    if j==3:
                        return np.sqrt(1./5)
                    elif j==2:
                        return np.sqrt(0.5)
                    elif j==1:
                        return np.sqrt(3./10)
                elif m1==0 and m2==0:
                    if j==3:
                        return np.sqrt(3./5)
                    elif j==1:
                        return -np.sqrt(2./5)
                elif m1==-1 and m2==1:
                    if j==3:
                        return np.sqrt(1./5)
                    elif j==2:
                        return -np.sqrt(1./2)
                    elif j==1:
                        return np.sqrt(3./10)
    else:
        return 0
                
                
    
def CG(l1,l2,l,m1,m2,m):
    prefactor = 1.0
    if m<0:
        prefactor *= -1.0**(l-l1-l2)
        m1*=-1
        m2*=-1
        m*=-1
    if l1<l2:
        prefactor *= -1.0**(l-l1-l2)
        tmp = l2
        l2 = l1
        l1 = tmp
        tmp = m2
        m2 = m1
        m1 = tmp
    return prefactor*CG_table(l1,l2,l,m1,m2,m)


if __name__=="__main__":
    t1 = dt.datetime.now()
    for i in range(1000):
        l = CG(1,0,1,0,1,1)
    t2 = dt.datetime.now()
    for i in range(1000):
        l2 = wig.clebsch_gordan(1,0,1,0,1,1)
    t3 = dt.datetime.now()
    del1 = t2-t1
    del2 = t3-t2
    delt = del2-del1
    print(del1)
    print(del2)
    print(delt)
    print(l2)
    print(l)
            
                
        
            