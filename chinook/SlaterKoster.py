#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Nov 13 19:26:02 2017

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
import chinook.wigner as Wlib
import chinook.Ylm as Ylm


def Vmat(l1,l2,V):
    '''
    For Slater-Koster matrix element generation, a potential matrix is
    sandwiched in between the two bond-rotating Dmatrices. It should be 
    of the shape 2*l1+1 x 2*l2+1, and have the V_l,l',D terms along the 
    'diagonal'-- a concept that is only well defined for a square matrix.
    For mismatched angular momentum channels, this turns into a diagonal 
    square matrix of dimension min(2*l1+1,2*l2+1) centred  along the larger
    axis. For channels where the orbital angular momentum change involves a
    change in parity, the potential should change sign, as per Slater Koster's
    original definition from 1954. This is taken care of automatically in 
    the Wigner formalism I use here, no need to have exceptions
    
    *args*:

        - **l1**, **l2**: int orbital angular momentum of initial and final states
        
        - **V**: numpy array of float -- length should be min(**l1** ,**l2**)*2+1
    
    *return*:

        - **Vm**: numpy array of float, shape 2 **l1** +1 x 2 **l2** +1
        
    ***
    '''
    if l1==0 or l2==0:
        Vm = np.zeros(max(2*l1+1,2*l2+1))
        Vm[int(len(Vm)/2)] = V[0]
    else:
        Vm = np.zeros((2*l1+1,2*l2+1))
        lmin = min(l1,l2)
        lmax = max(l1,l2)

        Vvals = np.identity(2*lmin+1)*np.array(V)

        if l2>l1:
            Vm[:,lmax-lmin:lmax-lmin+2*lmin+1] = Vvals
        else:
            Vm[lmax-lmin:lmax-lmin+2*lmin+1,:] = Vvals

    return np.atleast_2d(Vm)


def SK_cub(Ymats,l1,l2):
    '''
    In order to generate a set of independent Lambda functions for rapid 
    generation of Hamiltonian matrix elements, one must nest the 
    definition of the lambda functions within another function. In this way,
    we avoid cross-contamination of unrelated functions.
    The variables which are fixed for a given lambda function are the 
    cubic -to- spherical harmonics (Ymat) transformations, and the 
    orbital angular momentum of the relevant basis channels. The output
    lambda functions will be functions of the Euler-angles pertaining 
    to the hopping path, as well as the potential matrix V, which will be
    passed as a numpy array (min(l1,l2)*2+1) long of float.
    
    We follow the method described for rotated d-orbitals in the thesis of
    JM Carter from Toronto (HY Kee), where the Slater-Koster hopping
    matrix can be defined as the following operation:
        
        1. Transform local orbital basis into spherical harmonics
        2. Rotate the hopping path along the z-axis
        3. Product with the diagonal SK-matrix
        4. Rotate the path backwards
        5. Rotate back into basis of local orbitals
        6. Output matrix of hopping elements between all orbitals in the shell 
        to fill Hamiltonian
    
    *args*:

        - **Ymats**: list of numpy arrays corresponding to the relevant
        transformation from cubic to spherical harmonic basis
        
        - **l1**, **l2**: int orbital angular momentum channels relevant
        to a given hopping pair
        
    *return*:

        - lambda function for the SK-matrix between these orbital shells, 
        for arbitrary hopping strength and direction.
        
    ***
    '''
    def SK_build(EA,EB,Ey,V):
        o1rot = np.dot(Wlib.WignerD(l1,EA,EB,Ey),Ymats[0])
        o2rot = np.dot(Wlib.WignerD(l2,EA,EB,Ey),Ymats[1])
        try:
            return np.dot(np.conj(o1rot).T,np.atleast_2d(np.dot(Vmat(l1,l2,V),o2rot)))
        except ValueError:
            return np.dot(np.conj(o1rot).T,np.atleast_2d(np.dot(Vmat(l1,l2,V).T,o2rot)))

    return lambda EA,EB,Ey,V:SK_build(EA,EB,Ey,V)

def SK_full(basis):
    
    '''
    Generate a dictionary of lambda functions which take as keys the
    atom,orbital for both first and second element. 
    Formatting is a1a2n1n2l1l2, same as for SK dictionary entries
    
    *args*:

        - **basis**: list of orbital objects composing the TB-basis
    
    *return*:
    
        - **SK_funcs**: a dictionary of hopping matrix functions 
        (lambda functions with args EA,EB,Ey,V as Euler angles and potential (V))
        which can be executed for various hopping paths and potential strengths
        The keys of the dictionary will be organized similar to the way the SK
        parameters are passed, labelled by a1a2n1n2l1l2, which completely
        defines a given orbital-orbital coupling
        
    ***
    '''
    SK_funcs = {}
    Ymats = Ylm.Yproj(basis)
    for yi in Ymats:
        for yj in Ymats:
            if (yi[0],yj[0],yi[1],yj[1],yi[2],yj[2]) not in SK_funcs.keys():
                Y = [Ymats[yi],Ymats[yj]]
                SK_funcs[(yi[0],yj[0],yi[1],yj[1],yi[2],yj[2])] = SK_cub(Y,yi[2],yj[2])
    return SK_funcs
