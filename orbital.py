#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:21:15 2017

@author: ryanday

This is a rebuild of the rday_tblibrary2 intended to be more modular and clean than before.

The previous iteration was really founded around the idea of a Slater-Koster approach to the 
simulation, whereras in reality, this is often not the approach I take in these calculations.
Here I intend to instead have a Slater-Koster library which can be used to generate a Hamiltonian
matrix element list, but more generically, I will take any Hamiltonian list in the conventional form
as input.
This iteration is built off the basic centre of the simulation:
    
    Required input: hv,polarization,kmesh,eigenvectors,eigenenergies

The latter 2 will be the output of some Hamiltonian, of any form supported by the new software.


"""


import numpy as np

from sympy.physics.quantum.spin import Rotation
import electron_configs as econ
import atomic_mass as am
from operator import itemgetter


projdict={"0":np.array([[1.0,0.0,0.0,0.0]]),
               "1x":np.array([[-np.sqrt(0.5),0.0,1,1],[np.sqrt(0.5),0.0,1,-1]]),"1y":np.array([[0.0,np.sqrt(0.5),1,1],[0,np.sqrt(0.5),1,-1]]),"1z":np.array([[1,0,1,0]]),
                "2xy":np.array([[0.0,-np.sqrt(0.5),2,2],[0.0,np.sqrt(0.5),2,-2]]),"2yz":np.array([[0.0,np.sqrt(0.5),2,1],[0.0,np.sqrt(0.5),2,-1]]),
                "2xz":np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),"2ZR":np.array([[1.0,0.0,2,0]]),"2XY":np.array([[np.sqrt(0.5),0,2,2],[np.sqrt(0.5),0,2,-2]])}


hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23

class orbital:
    
    
    def __init__(self,atom,index,label,pos,Z,orient=None,spin=1,lam=0.0,sigma=1.0,slab_index=None):
        self.atom = atom #index of inequivalent atoms in basis
        self.index = index #index in the orbital basis
        self.label = label #label -- should be of format nlXX w/ lXX the orbital convention in projdict
        self.pos = pos #position in the lattice -- direct units in Angstrom
        self.Z = Z # atomic number
        self.mn = am.get_mass_from_number(self.Z)*mN #atomic mass
        self.spin = spin
        self.Dmat = None
        self.lam = lam
        self.sigma = 1.0
        self.n,self.l = int(self.label[0]),int(self.label[1])
        if slab_index==None:
            self.slab_index = index #this is redundant for bulk calc, but index in slab is distinct from lattice index
        else:
            self.slab_index = slab_index

        self.orient = orient
        if type(self.orient)==np.ndarray:
            self.proj = orient #projection into Ylm of form in projdict--array: [Real,Imag,l,ml]
        elif type(self.orient)==list: #can also pass a rotation from the conventional orientation
            self.proj = projdict[self.label[1:]]
            self.proj,self.Dmat = self.rot_projection(self.orient[0],self.orient[1])
        else:
            self.proj = projdict[self.label[1:]]
            
    def copy(self):
        
        return orbital(self.atom,self.index,self.label,self.pos,self.Z,self.orient,self.spin,self.lam,self.sigma,self.slab_index)
#    
#    def copyslab(self):    
#        return orbital(self.atom,self.index,self.label,self.pos,self.Z,self.orient,self.spin,self.lam,self.sigma,self.slab_index)

        
    def rot_projection(self,vector,gamma):
        
        '''
        Go through the projection array, and apply the correct transformation to
        the Ylm projections in order
        Define Euler angles in the z-y-z convention
        THIS WILL BE A COUNTERCLOCKWISE ROTATION ABOUT vector BY gamma
        '''
        vector = vector/np.linalg.norm(vector)
        Ylm_vec = np.zeros((2*self.l+1),dtype=complex)
        for a in range(len(self.proj)):
            Ylm_vec[int(self.proj[a,-1]+self.l)] +=self.proj[a,0]+1.0j*self.proj[a,1]
        
        A,B,y = Euler(vector,gamma)
        Dmat = Dmatrix(self.l,A,B,y)
        
        
        Ynew = np.dot(np.conj(Dmat),Ylm_vec)
        proj = []

        for a in range(2*self.l+1):
            if abs(Ynew[a])>10**-10:
                proj.append([np.around(np.real(Ynew[a]),10),np.around(np.imag(Ynew[a]),10),self.l,a-self.l])
                
        proj = np.array(proj)
        return proj,Dmat

def fact(n):
    if n<0:
        print('negative factorial!')
        return 0
    elif n==0:
        return 1
    else:
        return n*fact(n-1)
    
def Rmat(n,t):
    '''
    Rodrigues theorem for rotations. 
    args:
        n --> np.array x 3 axis of rotation
        t --> float radian angle of rotation counter clockwise for t>0
    return:
        R --> np.array (3x3) of float rotation matrix
    '''
    n = n/np.linalg.norm(n)
    K = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    R = np.identity(3)+np.sin(t)*K+(1-np.cos(t))*np.dot(K,K)
    return R   
    
def Euler(n,t):
    '''
    Extract the Euler angles for a ZYZ rotation around n by t
    args: 
        n --> np.array x3 float for axis
        t --> float radian angle of rotation counter clockwise for t>0
    Has special case for B = +/- Z*pi where conventional approach doesn't work due to division by zero
    '''
    R = Rmat(n,t)

    b = np.arccos(R[2,2])
    sb = np.sin(b)
    a,y=0.0,0.0
    if abs(sb)>10.0**-6:
        a = np.arctan2(R[2,1]/sb,-R[2,0]/sb)
        y = np.arctan2(R[1,2]/sb,R[0,2]/sb)
    else:
        a=[np.arctan2(R[1,0],R[0,0]),np.arctan2(R[1,0],-R[0,0])]

        if R[2,2]<0:
            a = np.arctan2(R[1,0],R[0,0])

        else:
            a = np.arctan2(R[1,0],-R[0,0])

    return a,b,y

def Dmatrix(l,A,B,y):
    Dmat = np.zeros((2*l+1,2*l+1),dtype=complex)
    for m_i in range(2*l+1):
        for mp_i in range(2*l+1):
            m = m_i-l
            mp = mp_i-l
            Dmat[mp_i,m_i] = np.conj(Rotation.D(l,mp,m,y,B,A).doit())
    return Dmat


def sort_basis(base,slab):
    '''
    Utility script for organizing an orbital basis that is out of sequence
    args: base -- list of orbital objects
            slab -- bool, True or False if this is for sorting a slab
    return: org_base -- list or sorted orbital objects (by orbital.index value)
    '''
    inds = [[b[0],b[1].index] for b in list(enumerate(base))]
    
    ind_sort = sorted(inds,key=itemgetter(1))
#    if slab:
#        orb_base = [base[i[0]].copyslab() for i in ind_sort]
#    else:
    orb_base = [base[i[0]].copy() for i in ind_sort]
    
    return orb_base
    
def spin_double(basis,lamdict):
    '''
    Double the size of a basis to introduce spin to the problem.
    Go through the basis and create an identical copy with opposite spin and 
    incremented index such that the orbital basis order puts spin down in the first
    N/2 orbitals, and spin up in the second N/2.
    Args: basis -- list of orbital objects
        lamdict -- spin-orbit coupling_strength for the different atomic species
    
    return: doubled basis carrying all necessary spin information
    '''
    LB = len(basis)
    b_2 = []
    for ind in range(LB):
        basis[ind].spin = -1
        basis[ind].lam = lamdict[basis[ind].atom]
        spin_up = basis[ind].copy()
        spin_up.spin = 1
        spin_up.index = basis[ind].index+LB
        b_2.append(spin_up)
    return basis + b_2
        

if __name__=="__main__":
    
    Z = 26
    i = 0
    label = ["32xz","32yz","32xy"]
    pos = np.zeros(3)
    
    dxz = orbital(0,i,label[0],pos,Z)
    dyz = orbital(0,i+1,label[1],pos,Z)
    dxy = orbital(0,i+2,label[2],pos,Z)
    
#    v = np.array([1,0,0]) #### STILL NOT QUITE RIGHT, GIVING DIFFERENT CHIRALITIES FOR  x AND z rotation
#    a = np.pi/2
#    print('x')
#    print(dxz.proj)
#    proj_x = dxz.rot_projection(v,a)
#    print(proj_x)
  
        
                
        
        
        
        
        
        
        
        

            
        