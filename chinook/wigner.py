# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:54:44 2018

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
from sympy.physics.quantum.spin import Rotation
import chinook.rotation_lib as rotlib
import chinook.plt_sph_harm as sph
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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



def symm_D(l,A,B,y):
    '''
    For testing purposes only, generate the Wigner Big D matrix using the 
    sympy package, for comparison.
    
    *args*:
        - **l**: int (or half integer) angular momentum
        
        - **Euler_A**, **Euler_B**, **Euler_y**: float z-y-z Euler angles defining
        the rotation
    
    *return*:
        - **Dmat**: 2j+1 x 2j+1 numpy array of complex float
    
    ***
    '''
    Dmat = np.zeros((int(2*l+1),int(2*l+1)),dtype=complex)
    for m_i in range(int(2*l+1)):
        for mp_i in range(int(2*l+1)):
            m = m_i-l
            mp = mp_i-l
            Dmat[int(mp_i),int(m_i)] = Rotation.D(l,mp,m,Euler_A,Euler_B,Euler_y).doit()
    return Dmat


def testing(j,Euler_A,Euler_B,Euler_y):
    '''
    Test function comparing rotation via Wigner matrices here, against the
    D matrices defined in the sympy library.
    
    *args*:
        - **j**: (half-) int, orbital angular momentum
        
        - **Euler_A**, **Euler_B**, **Euler_y**: float z-y-z Euler angles defining
        the rotation
        
    '''
    sym_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    my_D = np.zeros((2*j+1,2*j+1),dtype=complex)
    t1 = dt.datetime.now()
    Do = symm_D(j,Euler_A,Euler_B,Euler_y)
    Dp = WignerD(j,Euler_A,Euler_B,Euler_y)

    for m in range(-j,j+1):
        for mp in range(-j,j+1):
            sym_D[mp+j,m+j] = Rotation.D(j,mp,m,Euler_A,Euler_B,Euler_y).doit()
    t2 = dt.datetime.now()
    for m in range(-j,j+1):
        for mp in range(-j,j+1):
            my_D[mp+j,m+j] = big_D_element(j,mp,m,Euler_A,Euler_B,Euler_y)
    t3 = dt.datetime.now()
    
    print('sympy:',t2-t1)
    print('rpd:',t3-t2)
    
    
    print('auto_symm minus auto Wig')        
    print(np.linalg.norm(Do-Dp))
    print('man sym - man wig')
    print(np.linalg.norm(sym_D-my_D))
    
    
def rotate_orbital_test(projection,rotation):
    '''
    Diagnostic test for the rotation of orbitals using the Wigner matrices
    defined here.
    
    *args*:
        - **projection**: numpy array of Nx4 float indicating orbital projection
        in spherical harmonics. Each row being [Re(proj), Im(proj), l,m]
        
        - **rotation**: numpy array of 3x3 float, or list of len 3 numpy array float, and
        angle (float).
        
    '''
    l = int(projection[0,2])
    Ylm_vec = np.zeros((2*l+1),dtype=complex)
    for a in range(len(projection)):
        Ylm_vec[int(projection[a,-1]+l)] +=projection[a,0]+1.0j*projection[a,1]
    
    if type(rotation)==np.ndarray:
        Rmatrix = rotation
    else:
        print('the type is',type(rotation))
        Rmatrix = rotlib.Rodrigues_Rmat(*rotation)
        print(np.around(Rmatrix,3))
    A,B,y = rotlib.Euler(Rmatrix)
    
    Dmat = WignerD(l,A,B,y)
    Ynew = np.dot(Dmat,Ylm_vec)

    proj = []

    for a in range(2*l+1):
        if abs(Ynew[a])>10**-10:
            proj.append([np.around(np.real(Ynew[a]),10),np.around(np.imag(Ynew[a]),10),l,a-l])
                
    proj = np.array(proj)
    x0,y0,z0,tri0,cols0 = sph.plot_orbital(20,projection)

    x1,y1,z1,tri1,cols1 = sph.plot_orbital(20,proj)
    
    
    fig = plt.figure()
    ax0 = fig.add_subplot(121,projection='3d')
    ax1 = fig.add_subplot(122,projection='3d')
    p = ax0.plot_trisurf(x0,y0,z0,triangles=tri0,cmap=cm.RdBu,shade=True,antialiased=True,edgecolors='w',linewidth=0.1)

    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    p.set_array(cols0)
    p.set_clim(-0.7,0.7)
    
    p1 = ax1.plot_trisurf(x1,y1,z1,triangles=tri1,cmap=cm.RdBu,shade=True,antialiased=True,edgecolors='w',linewidth=0.1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    p1.set_array(cols1)
    p1.set_clim(-0.7,0.7)
    if l==1:
        Yp = np.array([[0.707,0.707j,0],[0,0,1],[-0.707,0.707j,0]])
        p0Y = np.zeros(3,dtype=complex)
        pfY = np.zeros(3,dtype=complex)
        for vi in range(len(projection)):
            p0Y[int(projection[vi,-1]+1)]+=projection[vi,0] + projection[vi,1]*1.0j
        for vi in range(len(proj)):
            pfY[int(proj[vi,-1]+1)]+=proj[vi,0] + proj[vi,1]*1.0j
        p0C = np.array([np.zeros(3),np.dot(np.linalg.inv(Yp),p0Y)])
        pfC = np.array([np.zeros(3),np.dot(np.linalg.inv(Yp),pfY)])
        print('Vector Before: ',np.around(p0C[1],3))
        print('Vector After: ',np.around(pfC[1],3))
        print('Actual Vector After: ',np.around(np.dot(p0C[1],Rmatrix),3))
    
    ax0.plot(p0C[:,0],p0C[:,1],p0C[:,2],c='r')
    ax1.plot(pfC[:,0],pfC[:,1],pfC[:,2],c='r')
    
    
#if __name__ == "__main__":
    
#    A,B,y = (-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random()),(-np.pi+2*np.pi*np.random.random())
#    A,B,y = 2.3562,0.95532,0.523599
#    j = 2
#    testing(j,A,B,y)
    
#    o1 = np.array([[0.707,0,1,-1],[-0.707,0,1,1]])
#    o2 = np.array([[0,0.707,1,1],[0,0.707,1,-1]])
#    o3 = np.array([[1,0,1,0]])
#    v1 = np.array([0,0,1])
#    g = np.pi*2/3
#    R = np.array([[-0.70710678,  0.40824829,  0.57735027],
#       [-0.        , -0.81649658,  0.57735027],
#       [ 0.70710678,  0.40824829,  0.57735027]])
##    
##    R = np.array([[-0.5      ,  0.8660254,  0.       ],
##       [-0.8660254, -0.5      ,  0.       ],
##       [ 0.       ,  0.       ,  1.       ]])
#    rotate_orbital_test(o2,R)

    