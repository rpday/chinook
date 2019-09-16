#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Thu Nov  9 21:21:15 2017
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
import chinook.atomic_mass as am
from chinook.wigner import WignerD
import chinook.rotation_lib as rotlib
from operator import itemgetter


##STANDARD ORBITAL PROJECTIONS IN THE BASIS OF SPHERICAL HARMONICS Y_LM
projdict={"0":np.array([[1.0,0.0,0.0,0.0]]),
               "1x":np.array([[-np.sqrt(0.5),0.0,1,1],[np.sqrt(0.5),0.0,1,-1]]),"1y":np.array([[0.0,np.sqrt(0.5),1,1],[0,np.sqrt(0.5),1,-1]]),"1z":np.array([[1,0,1,0]]),
                "2xy":np.array([[0.0,-np.sqrt(0.5),2,2],[0.0,np.sqrt(0.5),2,-2]]),"2yz":np.array([[0.0,np.sqrt(0.5),2,1],[0.0,np.sqrt(0.5),2,-1]]),
                "2xz":np.array([[-np.sqrt(0.5),0,2,1],[np.sqrt(0.5),0,2,-1]]),"2ZR":np.array([[1.0,0.0,2,0]]),"2XY":np.array([[np.sqrt(0.5),0,2,2],[np.sqrt(0.5),0,2,-2]]),
                "3z3":np.array([[1.0,0.0,3,0]]),"3xz2":np.array([[np.sqrt(0.5),0,3,-1],[-np.sqrt(0.5),0,3,1]]),
                "3yz2":np.array([[0,np.sqrt(0.5),3,-1],[0,np.sqrt(0.5),3,1]]),"3xzy":np.array([[0,-np.sqrt(0.5),3,2],[0,np.sqrt(0.5),3,-2]]),
                "3zXY":np.array([[np.sqrt(0.5),0,3,2],[np.sqrt(0.5),0,3,-2]]),"3xXY":np.array([[-np.sqrt(0.5),0,3,3],[np.sqrt(0.5),0,3,-3]]),
                "3yXY":np.array([[0,np.sqrt(0.5),3,3],[0,np.sqrt(0.5),3,-3]])}
                

##PHYSICAL CONSTANTS##
hb = 6.626*10**-34/(2*np.pi)
c  = 3.0*10**8
q = 1.602*10**-19
A = 10.0**-10
me = 9.11*10**-31
mN = 1.67*10**-27
kb = 1.38*10**-23

class orbital:
    '''
    The **orbital** object carries all essential details of the elements of the
    model Hamiltonian basis, for both generation of the tight-binding model, in
    addition to the evaluation of expectation values and ARPES intensity.
    
    '''
    
    def __init__(self,atom,index,label,pos,Z,orient=[0.0],spin=1,lam=0.0,sigma=1.0,slab_index=None):
        
        '''
        Initialize the **orbital** object.
        
        *args*:

            - **atom**: int, index of inequivalent atoms in the basis
            
            - **index**: int, index in the list of orbitals which consitute the 
            basis
            
            - **label**: string, should be of format nlXX w/ lXX the orbital convention at top of *orbital.py*
        
            - **pos**: numpy array of 3 float, position of orbital in direct space, units of Angstrom
            
            - **Z**: int, atomic number
            
        *kwargs*:

            - **orient**: various possible format. Can pass a projection array, 
            as in case of **projdict**, each element being length 4 array of 
            [Re(proj),Im(proj), l,m]. Alternatively, can pass a conventional 
            orbital label, and define a rotation: this is passed as a list, with 
            first element a numpy array of 3 float indicating the rotation axis, 
            and the second element a float, corresponding to angle, in radian,
            of the rotation
                
            - **spin**: int, +/-1, indicating the spin projection, default to 
            1 for spinless calculations
                
            - **lam**: float, strength of spin-orbit coupling, eV
                
            - **sigma**: float, effective scattering cross section, a.u.
                
            - **slab_index**: int, index in ordered list of slab orbitals, unused for bulk
            calculations
            
        ***
        '''
        
        
        self.atom = atom
        self.index = index
        self.label = label
        self.pos = pos 
        self.Z = Z 
        self.mn = am.get_mass_from_number(self.Z)*mN #atomic mass
        self.spin = spin #spin (+/-1)
        self.lam = lam #spin-orbit coupling strength
        self.sigma = sigma #scattering cross section, can be reduced/amplified for a given orbital
        self.n,self.l = int(self.label[0]),int(self.label[1])
        self.Dmat = np.identity(2*self.l+1)
        if slab_index is None:
            self.slab_index = index #this is redundant for bulk calc, but index in slab is distinct from lattice index
            self.depth = 0.0
        else:
            self.slab_index = slab_index
            self.depth = self.pos[2]

        self.orient = orient
        if type(self.orient)==np.ndarray:
            self.proj = self.orient #projection into Ylm of form in projdict--array: [Real,Imag,l,ml]
        elif type(self.orient)==list: #can also pass a rotation from the conventional orientation
            self.proj = projdict[self.label[1:]]
            if abs(self.orient[-1])>0:
                self.proj,self.Dmat = rot_projection(self.l,self.proj,self.orient)

        else:
            self.proj = projdict[self.label[1:]]
            
    def copy(self):

        '''
        Copy by value method for orbital object
        
        *return*:

            - **orbital_copy**: duplicate of **orbital** object
            
        ***
        '''
        orbital_copy = orbital(self.atom,self.index,self.label,self.pos,self.Z,self.orient,self.spin,self.lam,self.sigma,self.slab_index)
        orbital_copy.proj = self.proj.copy()
        orbital_copy.Dmat = self.Dmat.copy()
        return orbital_copy

        
def rot_projection(l,proj,rotation):
    
    '''
    Go through a projection array, and apply the intended transformation to
    the Ylm projections in order.
    Define Euler angles in the z-y-z convention
    THIS WILL BE A COUNTERCLOCKWISE ROTATION ABOUT a vector BY angle gamma 
    expressed in radians. Note that we always define spin in the lab-frame, so
    spin degrees of freedom are not rotated when we rotate the orbital degrees
    of freedom.
    
    *args*:

        - **l**: int,orbital angular momentum
        
        - **proj**: numpy array of shape Nx4 of float, each element is
        [Re(projection),Im(projection),l,m]
        
        - **rotation**: float, or list, defining rotation of orbital. If float,
        assume rotation about z-axis. If list, first element is a numpy array 
        of len 3, indicating rotation vector, and second element is float, angle.
        
    *return*:

        - **proj**: numpy array of Mx4 float, as above, but modified, and may
        now include additional, or fewer elements than input *proj*.

        - **Dmat**: numpy array of (2l+1)x(2l+1) complex float indicating the 
        Wigner Big-D matrix associated with the rotation of this orbital shell
        about the intended axis.
    
    ***
    '''
        
    if len(rotation)==1:
        rotation = (np.array([0,0,1]),rotation)
    Ylm_vec = np.zeros((2*l+1),dtype=complex)
    for a in range(len(proj)):
        Ylm_vec[int(proj[a,-1]+l)] +=proj[a,0]+1.0j*proj[a,1]
    
    A,B,y = rotlib.Euler(rotation)
    Dmat = WignerD(l,A,B,y)
    Ynew = np.dot(Dmat,Ylm_vec)

    proj = []

    for a in range(2*l+1):
        if abs(Ynew[a])>10**-10:
            proj.append([np.around(np.real(Ynew[a]),10),np.around(np.imag(Ynew[a]),10),l,a-l])
            
    proj = np.array(proj)
    return proj,Dmat

def fact(n):

    '''
    Recursive factorial function, works for any non-negative integer.
    
    *args*:

        - **n**: int, or integer-float
        
    *return*:

        - int, recursively evaluates the factorial of the initial input value.
    
    ***
    '''
    if (np.floor(n)-n)>0:
        print('Error, non-integer value passed to factorial.')
        return 0
    if n<0:
        print('negative factorial!')
        return 0
    elif n==0:
        return 1
    else:
        return n*fact(n-1)
    
    
def slab_basis_copy(basis,new_posns,new_inds):
    
    '''
    Copy elements of a slab basis into a new list of
    orbitals, with modified positions and index ordering.
    
    *args*:

        - **basis**: list or orbital objects
        
        - **new_posns**: numpy array of len(basis)x3 float, new positions for
        orbital
        
        - **new_inds**: numpy array of len(basis) int, new indices for orbitals
        
    *return*:

        - **new_basis**: list of duplicated orbitals following modification.
    
    ***
    '''
    new_basis = np.empty(len(basis),dtype=orbital)
    for o in list(enumerate(basis)):
        local_orb_copy = o[1].copy()
    
        
        local_orb_copy.slab_index = int(new_inds[o[0]])
        local_orb_copy.index = o[0]
        local_orb_copy.pos = new_posns[o[0]]
        new_basis[int(new_inds[o[0]])] = local_orb_copy
    return new_basis
    

def sort_basis(basis,slab):
    
    '''
    Utility script for organizing an orbital basis that is out of sequence
    
    *args*:

        - **basis**: list of orbital objects
        
        - **slab**: bool, True or False if this is for sorting a slab
        
    *return*:

        - **orb_basis**: list of sorted orbital objects (by orbital.index value)
        
    ***
    '''
    inds = [[b[0],b[1].index] for b in list(enumerate(basis))]
    
    ind_sort = sorted(inds,key=itemgetter(1))

    orb_basis = [basis[i[0]].copy() for i in ind_sort]
    
    return orb_basis
    
def spin_double(basis,lamdict):

    '''
    Double the size of a basis to introduce spin to the problem.
    Go through the basis and create an identical copy with opposite spin and 
    incremented index such that the orbital basis order puts spin down in the first
    N/2 orbitals, and spin up in the second N/2.
    
    *args*:

        - **basis**: list of orbital objects
       
        - **lamdict**: dictionary of int:float pairs providing the
        spin-orbit coupling strength for the different inequivalent atoms in 
        basis.
    
    *return*:
    
        - doubled basis carrying all required spin information
        
    ***
    '''
    LB = len(basis)
    basis_double = []

    for ind in range(LB):
        basis[ind].spin = -1
        basis[ind].lam = lamdict[basis[ind].atom]
        spin_up = basis[ind].copy()
        spin_up.spin = 1
        spin_up.index = basis[ind].index+LB
        basis_double.append(spin_up)
    return basis + basis_double

            
if __name__=="__main__":
    
    Z = 26
    i = 0
    label = ["32xz","32yz","32xy"]
    pos = np.zeros(3)
    
    dxz = orbital(0,i,label[0],pos,Z)
    dyz = orbital(0,i+1,label[1],pos,Z)
    dxy = orbital(0,i+2,label[2],pos,Z)
    
    v = np.array([1,0,0]) 
    a = np.pi/2
    print('x')
    print(np.around(dxz.proj,3))
    proj_x,_ = rot_projection(2,dxz.proj,[v,a])
    print(np.around(proj_x,3))
  
        