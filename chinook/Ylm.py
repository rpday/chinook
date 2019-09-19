#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Sun Jun 18 15:46:21 2017

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
from math import factorial



projdict={"0":np.array([[1.0,0.0,0.0,0.0]]),
               "1x":np.array([[-np.sqrt(0.5),0.0,1,1],[np.sqrt(0.5),0.0,1,-1]]),"1y":np.array([[0.0,np.sqrt(0.5),1,1],[0,np.sqrt(0.5),1,-1]]),"1z":np.array([[1,0,1,0]]),
                "2xy":np.array([[0.0,-np.sqrt(0.5),2,2],[0.0,np.sqrt(0.5),2,-2]]),"2yz":np.array([[0.0,np.sqrt(0.5),2,1],[0.0,np.sqrt(0.5),2,-1]]),
                "2xz":np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),"2ZR":np.array([[1.0,0.0,2,0]]),"2XY":np.array([[np.sqrt(0.5),0,2,2],[np.sqrt(0.5),0,2,-2]]),
                "3z3":np.array([[1.0,0.0,3,0]]),"3xz2":np.array([[np.sqrt(0.5),0,3,-1],[-np.sqrt(0.5),0,3,1]]),
                "3yz2":np.array([[0,np.sqrt(0.5),3,-1],[0,np.sqrt(0.5),3,1]]),"3xzy":np.array([[0,-np.sqrt(0.5),3,2],[0,np.sqrt(0.5),3,-2]]),
                "3zXY":np.array([[np.sqrt(0.5),0,3,2],[np.sqrt(0.5),0,3,-2]]),"3xXY":np.array([[-np.sqrt(0.5),0,3,3],[np.sqrt(0.5),0,3,-3]]),
                "3yXY":np.array([[0,np.sqrt(0.5),3,3],[0,np.sqrt(0.5),3,-3]])}
          

def Y(l,m,theta,phi):
    '''
    Spherical harmonics, defined here up to l = 4. This allows for photoemission from
    initial states up to and including f-electrons (final states can be d- or g- like).
    Can be vectorized with numpy.vectorize() to allow array-like input
    
    *args*:

        - **l**: int orbital angular momentum, up to l=4 supported
        
        - **m**: int, azimuthal angular momentum |m|<=l
        
        - **theta**: float, angle in spherical coordinates, radian measured from the z-axis [0,pi]
        
        - **phi**: float, angle in spherical coordinates, radian measured from the x-axis [0,2pi]

    *return*:

        - complex float, value of spherical harmonic evaluated at theta,phi
    
    '''
    
    
    if l == 0:
        if m==0:
            return 0.5*np.sqrt(1.0/np.pi)*value_one(theta,phi)
        else:
            return 0.0 
               
    elif l == 1:
        if abs(m) == 1:
            return -np.sign(m)*0.5*np.sqrt(3.0/(2*np.pi))*np.exp(m*1.0j*phi)*np.sin(theta)
        elif m == 0:
            return 0.5*np.sqrt(3.0/np.pi)*np.cos(theta) 
        else:
            return 0.0

    elif l == 2:
        if abs(m) == 2:
            return 0.25*np.sqrt(15.0/(2*np.pi))*np.exp(m*1.0j*phi)*np.sin(theta)**2
        elif abs(m) == 1:
            return -np.sign(m)*0.5*np.sqrt(15.0/(2*np.pi))*np.exp(m*1.0j*phi)*np.sin(theta)*np.cos(theta)
        elif m==0:
            return 0.25*np.sqrt(5/np.pi)*(3*np.cos(theta)**2-1)
        else:
            return 0.0

    elif l == 3:
        if abs(m) == 3:
            return -np.sign(m)*1.0/8*np.sqrt(35/np.pi)*np.exp(m*1.0j*phi)*np.sin(theta)**3
        elif abs(m) == 2:
            return 1.0/4*np.sqrt(105/(2*np.pi))*np.exp(m*1.0j*phi)*np.sin(theta)**2*np.cos(theta)
        elif abs(m) == 1:
            return -np.sign(m)*1.0/8*np.sqrt(21/np.pi)*np.exp(m*1.0j*phi)*np.sin(theta)*(5*np.cos(theta)**2-1)
        elif m == 0:
            return 1.0/4*np.sqrt(7/np.pi)*(5*np.cos(theta)**3-3*np.cos(theta))
        else:
            return 0.0

    elif l == 4:
        if abs(m) == 4:
            return 3/16.*np.sqrt(35./2/np.pi)*np.sin(theta)**4*np.exp(m*1.0j*phi)
        elif abs(m) == 3:
            return -np.sign(m)*3/8.*np.sqrt(35/np.pi)*np.sin(theta)**3*np.cos(theta)*np.exp(m*1.0j*phi)
        elif abs(m) == 2:
            return 3./8.*np.sqrt(5/2/np.pi)*np.sin(theta)**2*(7*np.cos(theta)**2-1.0)*np.exp(m*1.0j*phi) 
        elif abs(m) == 1:
            return -np.sign(m)*3./8*np.sqrt(5/np.pi)*np.sin(theta)*(7*np.cos(theta)**3-3*np.cos(theta))*np.exp(m*1.0j*phi)
        elif m == 0:
            return 3./16.*np.sqrt(1./np.pi)*(35.*np.cos(theta)**4 - 30.*np.cos(theta)**2 + 3.0)
        else:
            return 0.0

    else:
        return 0.0
    
    
def value_one(theta,phi):
    '''
    Flexible generation of the number 1.0, in either float or array format

    *args*:

        - **theta**: float or numpy array of float

        - **phi**: float or numpy array of float

    *return*:

        **out**: float or numpy array of float, evaluated to 1.0, of same shape and type
        as **theta**, **phi**

    ***
    '''
    if type(theta)==np.ndarray:
        out =np.ones(np.shape(theta))
    if type(phi)==np.ndarray:
        out = np.ones(np.shape(phi))
    elif type(theta)!=np.ndarray and type(phi)!=np.ndarray:
        out = 1.0
    return out
        

def binom(a,b):
    '''
    Binomial coefficient for 'a choose b'

    *args*:

        - **a**: int, positive

        - **b**: int, positive

    *return*:

        - float, binomial coefficient

    ***
    '''
    return factorial(a+b)/float(factorial(a-b)*factorial(b))

def laguerre(x,l,j):
    '''
    Laguerre polynomial of order l, degree j, evaluated over x

    *args*:

        - **x**: float or numpy array of float, input

        - **l**: int, order of polynomial

        - **j**: int, degree of polynomial


    *return*:

        - **laguerre_output**: float or numpy array of float, shape as input **x**

    ***
    '''
    laguerre_output = sum([((-1)**i)*(binom(l+j,j-i)*x**i/float(factorial(i))) for i in range(j+1)])
    return laguerre_output


def gaunt(l,m,dl,dm):
    '''
    I prefer to avoid using the sympy library where possible, for speed reasons. These are the explicitly defined
    Gaunt coefficients required for dipole-allowed transitions (dl = +/-1) for arbitrary m,l and dm
    These have been tested against the sympy package to confirm numerical accuracy for all l,m possible
    up to l=5. This function is equivalent, for the subset of dm, dl allowed to
    sympy.physics.wigner.gaunt(l,1,l+dl,m,dm,-(m+dm))
    
    *args*:
        
        - **l**: int orbital angular momentum quantum number
        
        - **m**: int azimuthal angular momentum quantum number
        
        - **dl**: int change in l (+/-1)
        
        - **dm**: int change in azimuthal angular momentum (-1,0,1)
    
    *return*:

        - float Gaunt coefficient

    ***
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
    
    
def Yproj(basis):
    '''
    Define the unitary transformation rotating the basis of different inequivalent atoms in the
    basis to the basis of spherical harmonics for sake of defining L.S operator in basis of user
    
    29/09/2018 added reference to the spin character 'sp' to handle rotated systems effectively

    *args:*

        - **basis**: list of orbital objects
    
    *return*:

        - dictionary of matrices for the different atoms and l-shells--keys are tuples of (atom,l)
     
    ***
    '''
    normal_order = {0:{'':0},1:{'x':0,'y':1,'z':2},2:{'xz':0,'yz':1,'xy':2,'ZR':3,'XY':4},3:{'z3':0,'xz2':1,'yz2':2,'xzy':3,'zXY':4,'xXY':5,'yXY':6}}
    a = basis[0].atom
    n = basis[0].n
    l = basis[0].l
    sp = basis[0].spin
    M = {}
    M_tmp = np.zeros((2*l+1,2*l+1),dtype=complex)
    for b in basis:
        if np.linalg.norm(b.Dmat-np.identity(2*b.l+1))>0:
            Dmat = b.Dmat
        else:
            Dmat = None
        loc_rot = b.orient
        label = b.label[2:]

        if b.atom==a and b.n==n and b.l==l and b.spin==sp:
            for p in b.proj:
                M_tmp[l-int(p[-1]),normal_order[l][label]] = p[0]+1.0j*p[1]
                
        else:
                #If we are using a reduced basis, fill in orthonormalized projections for other states in the shell
                #which have been ignored in our basis choice--these will still be relevant to the definition of the LS operator
            M_tmp = fillin(M_tmp,l,Dmat)            
            M[(a,n,l,sp)] = M_tmp
                ##Initialize the next M matrix               
            a = b.atom
            n = b.n
            l = b.l
            sp = b.spin
            M_tmp = np.zeros((2*l+1,2*l+1),dtype=complex)
            for p in b.proj:
                M_tmp[l-int(p[-1]),normal_order[l][label]] = p[0]+1.0j*p[1]
    
    M_tmp = fillin(M_tmp,l,loc_rot)
    M[(a,n,l,sp)] = M_tmp
    
    return M

def fillin(M,l,Dmat=None):
    '''
    If only using a reduced subset of an orbital shell (for example, only t2g states in d-shell),
    need to fill in the rest of the projection matrix with some defaults

    *args*:

        - **M**: numpy array of (2l+1)x(2l+1) complex float

        - **l**: int

        - **Dmat**: numpy array of (2l+1)x(2l+1) complex float

    *return*:

        - **M**: numpy arrayof (2l+1)x(2l+1) complex float

    ***
    '''

    normal_order_rev = {0:{0:''},1:{0:'x',1:'y',2:'z'},2:{0:'xz',1:'yz',2:'xy',3:'ZR',4:'XY'},3:{0:'z3',1:'xz2',2:'yz2',3:'xzy',4:'zXY',5:'xXY',6:'yXY'}}

    for m in range(2*l+1):
        if np.linalg.norm(M[:,m])==0: #if column is empty (i.e. user-defined projection does not exist)
            proj = np.zeros(2*l+1,dtype=complex) 
            for pi in projdict[str(l)+normal_order_rev[l][m]]: 
                proj[l-int(pi[-1])] = pi[0]+1.0j*pi[1] #fill the column with generic projection for this orbital (this will be a dummy)
            if type(Dmat)==np.ndarray:
#                print('l: {:d},'.format(l),'Dmat: ',Dmat,'proj: ',proj)
                proj = np.dot(Dmat,proj)
            for mp in range(2*l+1): #Orthogonalize against the user-defined projections
                if mp!=m:
                    if np.linalg.norm(M[:,mp])!=0:
                        if np.dot(M[:,m],M[:,mp])>1e-10:
                            proj = GramSchmidt(proj,M[:,mp])
            M[:,m] = proj            
    return M
    

def GramSchmidt(a,b):
    '''
    Simple orthogonalization of two vectors, returns orthonormalized vector
    
    *args*:

        - **a**, **b**: numpy array of same length

    *returns*:

        - **GS_a**: numpy array of same size, orthonormalized to the b vector

    ***
    '''
    GS_a = a - np.dot(a,b)/np.dot(b,b)*b
    return GS_a/np.linalg.norm(GS_a)




    

    
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