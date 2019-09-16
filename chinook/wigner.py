# -*- coding: utf-8 -*-

#Created on Thu Sep 20 08:54:44 2018

#@author: ryanday
#MIT License

#Copyright (c) 2018 Ryan Patrick Day

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np

def fact(N):
    '''
    Recursive factorial function for non-negative integers.
    
    *args*:

        - **N**: int, or int-like float
        
    *return*:

        - factorial of **N**
        
    ***
    '''
    if abs(np.floor(N)-N)>0:
        return 0
    if N<0:
        return 0
    if N==0:
        return 1
    else:
        return N*fact(N-1)

def s_lims(j,m,mp):
    '''
    Limits for summation in definition of Wigner's little d-matrix
    
    *args*:

        - **j**: int,(or half-integer) total angular momentum quantum number
        
        - **m**: int, (or half-integer) initial azimuthal angular momentum quantum number
        
        - **mp**: int, (or half-integer) final azimuthal angular momentum 
        quantum number coupled to by rotation
        
    *return*:

        - list of int, viable candidates which result in well-defined factorials in 
        summation
    
    ***
    '''
    
    smin = max(0,m-mp)
    smax = min(j-mp,j+m)
    if smax<smin:
        smin,smax=0,0
    return np.arange(smin,smax+1,1)


def small_D(j,mp,m,Euler_B):
    '''
    Wigner's little d matrix, defined as         _
     j                                          \           (-1)^(mp-m+s)                 2j+m-mp-2s         mp-m+2s
    d    (B) = sqrt((j+mp)!(j-mp)!(j+m)!(j-m)!)  >  ----------------------------- cos(B/2)          sin(B/2)
     mp,m                                       /_s  (j+m-s)!s!(mp-m+s)!(j-mp-s)!
    
    where the sum over s includes all integers for which the factorial arguments in the denominator are non-negative.
    The limits for this summation are defined by s_lims(j,m,mp). 
    
    *args*:

        - **j**, **mp** ,**m** -- integer (or half-integer) angular momentum 
        quantum numbers for the orbital angular momentum, and its azimuthal projections
        which are related by the Wigner D matrix during the rotation
        
        - **Euler_B**: float, angle of rotation in radians, for the y-rotation
    
    *return*:

        - float representing the matrix element of Wigner's small d-matrix
    
    ***
    '''
    
    pref = np.sqrt(fact(j+mp)*fact(j-mp)*fact(j+m)*fact(j-m))
    cos = np.cos(Euler_B/2.)
    sin = np.sin(Euler_B/2.)
    s = s_lims(j,m,mp)
    s_sum = sum([(-1.0)**(mp-m+sp)/Wd_denominator(j,m,mp,sp)*cos**(2*j+m-mp-2*sp)*sin**(mp-m+2*sp) for sp in s])
    return pref*s_sum
    
    

def Wd_denominator(j,m,mp,sp):
    
    '''
    Small function for computing the denominator in the s-summation, one step 
    in defining the matrix elements of Wigner's small d-matrix
    
    *args*:

        - **j**, **m**, **mp**: integer (or half-integer) angular momentum 
        quantum numbers
        
        - **s**: int, the index of the summation
        
    *return*:

        - int, product of factorials
    
    ***
    '''
    return (fact(j+m-sp)*fact(sp)*fact(mp-m+sp)*fact(j-mp-sp))

def big_D_element(j,mp,m,Euler_A,Euler_B,Euler_y):
    '''
    Combining Wigner's small d matrix with the other two rotations, this defines
    Wigner's big D matrix, which defines the projection of an angular momentum state
    onto the other azimuthal projections of this angular momentum. Wigner defined 
    these matrices in the convention of a set of z-y-z Euler angles, passed here 
    along with the relevant quantum numbers:
        
    *args*: 

        - **j**, **mp**, **m**: integer (half-integer) angular momentum quantum numbers
        
        - **Euler_A**, **Euler_B**, **Euler_y**: float z-y-z Euler angles defining
        the rotation
        
    *return*:

        - complex float corresponding to the [mp,m] matrix element
    
    ***
    '''
    return np.exp(-1.0j*mp*Euler_A)*small_D(j,mp,m,Euler_B)*np.exp(-1.0j*m*Euler_y)


def WignerD(l,Euler_A,Euler_B,Euler_y):
    '''
    Full matrix representation of Wigner's Big D matrix relating the rotation
    of states within the subspace of the angular momentum l by the Euler rotation
    Z"(A)-Y'(B)-Z(y)
    
    *args*:

        - **l**: int (or half integer) angular momentum
        
        - **Euler_A**, **Euler_B**, **Euler_y**: float z-y-z Euler angles defining
        the rotation
    
    *return*:
    
        - **Dmat**: 2j+1 x 2j+1 numpy array of complex float
    
    ***
    '''
    Dmat = np.zeros((2*l+1,2*l+1),dtype=complex)
    for m_i in range(int(2*l+1)):
        for mp_i in range(int(2*l+1)):
            m=m_i-l
            mp=mp_i-l
            Dmat[int(m_i),int(mp_i)] = big_D_element(l,mp,m,Euler_A,Euler_B,Euler_y) 
    return Dmat


