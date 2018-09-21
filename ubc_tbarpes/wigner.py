# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:54:44 2018

@author: rday
"""

import numpy as np
import datetime as dt
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
    smin = max(0,m-mp)
    smax = min(j-mp,j+m)
    if smax<smin:
        smin,smax=0,0
    return np.arange(smin,smax+1,1)


def small_D(j,mp,m,B):
    '''
    Wigner's little d matrix, defined as         _
     j                                          \           (-1)^(mp-m+s)                 j+m-mp-2s         mp-m+2s
    d    (B) = sqrt((j+mp)!(j-mp)!(j+m)!(j-m)!)  >  ----------------------------- cos(B/2)          sin(B/2)
     mp,m                                       /_s  (j+m-s)!s!(mp-m+s)!(j-mp-s)!
    
    where the sum over s includes all integers for which the factorial arguments in the denominator are non-negative.
    The limits for this summation are defined by s_lims(j,m,mp). 
    args:
        j,mp,m -- integer (or half-integer) angular momentum quantum numbers for the orbital angular momentum, and its azimuthal projections
                  which are related by the Wigner D matrix during the rotation
        B -- float, angle of rotation in radians, for the y-rotation
    return:
        float representing the matrix element of Wigner's small d-matrix
    
    '''
    
    pref = np.sqrt(fact(j+mp)*fact(j-mp)*fact(j+m)*fact(j-m))
    cos = np.cos(B/2.)
    sin = np.sin(B/2.)
    s = s_lims(j,m,mp)
    s_sum = sum([(-1.0)**(mp-m+sp)/den(j,m,mp,sp)*cos**(2*j+m-mp-2*sp)*sin**(mp-m+2*sp) for sp in s])
    return pref*s_sum

def den(j,m,mp,sp):
    '''
    Small function for computing the denominator in the s-summation, one step in defining the matrix elements
    of Wigner's small d-matrix
    args:
        j,m,mp -- integer / half-integer angular momentum quantum numbers
        s -- the index of the summation
    return integer, product of factorials
    '''
    return (fact(j+m-sp)*fact(sp)*fact(mp-m+sp)*fact(j-mp-sp))

def big_D(j,mp,m,A,B,y):
    '''
    Combining Wigner's small d matrix with the other two rotations, this defines
    Wigner's big D matrix, which defines the projection of an angular momentum state
    onto the other azimuthal projections of this angular momentum. Wigner defined 
    these matrices in the convention of a set of z-y-z Euler angles, passed here 
    along with the relevant quantum numbers:
        
        args: 
            j,mp,m -- integer/half-integer angular momentum quantum numbers
            A,B,y -- float z-y-z Euler angles defining the rotation
        return:
            complex float corresponding to the [mp,m] matrix element
    '''
    return np.exp(-1.0j*mp*A)*small_D(j,mp,m,B)*np.exp(-1.0j*m*y)


def WignerD(l,A,B,y):
    '''
    Full matrix representation of Wigner's Big D matrix relating the rotation
    of states within the subspace of the angular momentum l by the Euler rotation
    Z"(A)-Y'(B)-Z(y)
    args:
        l -- integer (or half integer) angular momentum
        A,B,y -- float, radians z-y-z convention Euler angles
    return:
        Dmat -- 2j+1 x 2j+1 numpy array of complex float
    '''
    Dmat = np.zeros((2*l+1,2*l+1),dtype=complex)
    for m_i in range(int(2*l+1)):
        for mp_i in range(int(2*l+1)):
            m=m_i-l
            mp=mp_i-l
            Dmat[int(mp_i),int(m_i)] = big_D(l,mp,m,A,B,y) 
    return Dmat

def symm_D(l,A,B,y):
    Dmat = np.zeros((int(2*l+1),int(2*l+1)),dtype=complex)
    for m_i in range(int(2*l+1)):
        for mp_i in range(int(2*l+1)):
            m = m_i-l
            mp = mp_i-l
            Dmat[int(mp_i),int(m_i)] = Rotation.D(l,mp,m,A,B,y).doit()
    return Dmat


def testing(j,A,B,y):
    '''
    Test function comparing
    '''
    sym_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    my_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    t1 = dt.datetime.now()
    Do = symm_D(j,A,B,y)
    Dp = WignerD(j,A,B,y)

    for m in range(-j,j+1):
        for mp in range(-j,j+1):
            sym_D[mp+j,m+j] = Rotation.D(j,mp,m,A,B,y).doit()
    t2 = dt.datetime.now()
    for m in range(-j,j+1):
        for mp in range(-j,j+1):
            my_D[mp+j,m+j] = big_D(j,mp,m,A,B,y)
    t3 = dt.datetime.now()
    
    print('sympy:',t2-t1)
    print('rpd:',t3-t2)
    
    
    print('auto_symm minus auto Wig')        
    print(np.linalg.norm(Do-Dp))
    print('man sym - man wig')
    print(np.linalg.norm(sym_D-my_D))
    
if __name__ == "__main__":
    
    A,B,y = (-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random())
    A,B,y = 2.3562,0.95532,0.523599
    j = 2
    testing(j,A,B,y)

    