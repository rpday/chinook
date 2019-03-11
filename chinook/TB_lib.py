#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:41:01 2017

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
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import compress
import sys

if sys.version_info<(3,0):
    print('Warning: This software requires Python 3.0 or higher. Please update your Python instance before proceeding')
else:
    import chinook.H_library as Hlib


'''
Tight-Binding Utility module

'''


class H_me:
    '''
    This class contains the relevant executables and data structure pertaining 
    to generation of the Hamiltonian matrix elements for a single set of 
    coupled basis orbitals. Its attributes include integer values 
    **i**, **j** indicating the basis indices, and a list of hopping
    vectors/matrix element values for the Hamiltonian. 
    
    The method **H2Hk** provides an executable function of momentum to allow
    broadcasting of the Hamiltonian over a large array of momenta.
    Python's flexible protocol for equivalency and passing variables by
    reference/value require definition of a copy operator which allows one to
    produce safely, a copy of the object rather than its coordinates 
    in memory alone.
    ***
    '''
    def __init__(self,i,j):
        '''
        Initialize the H_me object, the related basis indices and an empty list
        of hopping elements.
        
        *args*:
            - **i**, **j**: int, basis indices
            
        ***
        '''
        self.i = i
        self.j = j
        self.H = []
    
    def append_H(self,R0,R1,R2,H):
        '''
        Add a new hopping path to the coupling of the parent orbitals.
        
        *args*:
            - **R0**, **R1**, **R2**: float connecting vector in cartesian
            coordinate frame--this is the TOTAL vector, not the relevant 
            lattice vectors only
            
            - **H**: complex float, matrix element strength
            
        *return*:
            - directly modifies the Hamiltonian list for these matrix
            coordinates
        '''
        
        self.H.append([R0,R1,R2,H])
        
        
    def H2Hk(self):  
        '''
        Transform the list of hopping elements into a Fourier-series expansion 
        of the Hamiltonian. This is run during diagonalization for each
        matrix element index
        
        *return*:
            - lambda function of a numpy array of float of length 3
            
        ***
        '''
        return lambda x: sum([complex(m[-1])*np.exp(1.0j*np.dot(x,np.array([m[0],m[1],m[2]]))) for m in self.H])
        
    def clean_H(self):
        
        '''
        
        Remove all duplicate instances of hopping elements in the matrix 
        element list. This function is run automatically during slab generation.
        
        The Hamiltonian list is not itself directly modified. 
        
        *return*:
            - list of hopping vectors and associated Hamiltonian matrix
            element strengths
            
        ***
        '''
        Hel_double = self.H
        bools = [True]*len(Hel_double)
        for hi in range(len(Hel_double)-1):
            for hj in range(hi+1,len(Hel_double)):
                
                try:
                    norm = np.linalg.norm(np.array(Hel_double[hi])-np.array(Hel_double[hj]))
                    if abs(norm)<1e-5:
                        bools[hj] = False
                except IndexError:
                    continue
        return list(compress(Hel_double,bools))
    
    def copy(self):
        
        '''
        Copy by value of the **H_me** object

        *return*:
            - **H_copy**: duplicate **H_me** object
        ***
        '''
        
        H_copy = H_me(self.i,self.j)
        H_copy.H = [h for h in self.H]
        return H_copy
    
    
        
class TB_model:
    '''
    The **TB_model** object carries the model basis as a list of **orbital**
    objects, as well as the model Hamiltonian, as a list of **H_me**.
    '''
    
    def __init__(self,basis,H_args,Kobj = None):
        '''
        Initialize the tight-binding model object.
        
        *args*:   
            - **basis**: list of orbital objects
                    
            - **H_args**: dictionary for Hamiltonian build:
            
                - *'type'*: string,  "SK" ,"list", or "txt"
             
            - if *'type'* is *'SK'*:
                - *'V'*: dictionary of Slater-Koster hopping terms, 
                c.f. **chinook.H_library**
                
                - *'avec'*: numpy array of 3x3 float,
                lattice vectors for generating neighbours
                 
                
             - if *'type'* is *'list'*:
                 
                 - *'list'*: list of Hamiltonian elements: [i,j,R1,R2,R3,Hij]
                 
             - if *'type'* is 'txt':
                 
                 - *'filename'*:path to text file
                 
             - *'cutoff'*: float, cutoff distance for hopping
             
             - *'renorm'*: float, renormalization factor
                 
             - *'offset'*: float, offset energy, in eV
             
             - *'tol'*: float, minimum amplitude for matrix element term
            
             - *'spin'* : dictionary, with key values of 'bool':boolean,
             'soc':boolean, 'lam': dictionary of integer:float pairs
        
        *kwargs*:
            - **Kobj**: momentum object, as defined in *chinook.klib.py*
            
        ***
        '''
        self.basis = basis 
    
        if H_args is not None:
            if 'avec' in H_args.keys():
                self.avec = H_args['avec']
        
            self.mat_els = self.build_ham(H_args)
            
        self.Kobj = Kobj

        
    def copy(self):
        '''
        Copy by value of the **TB_model** object
        
        *return*:
            - **TB_copy**: duplicate of the **TB_model** object.
        '''
        
        basis_copy = [o.copy() for o in self.basis]
        TB_copy = TB_model(basis_copy,None,self.Kobj)
        TB_copy.avec = self.avec.copy()
        TB_copy.mat_els = [m.copy() for m in self.mat_els]
        
        return TB_copy
        
            
    
    def build_ham(self,H_args):
        
        '''
        Buld the Hamiltonian using functions from **chinook.H_library.py**

        *args*:
            - **H_args**: dictionary, containing all relevant information for
            defining the Hamiltonian list. For details, see **TB_model.__init__**.
            
        *return*:
            - sorted list of matrix element objects. These objects have 
            i,j attributes referencing the orbital basis indices, 
            and a list of form [R0,R1,R2,Re(H)+1.0jIm(H)]
        
        ***
        '''
        if type(H_args)==dict:
            try:
                ham_list = []
                if H_args['type'] == "SK":
                    ham_list = Hlib.sk_build(H_args['avec'],self.basis,H_args['V'],H_args['cutoff'],H_args['tol'],H_args['renorm'],H_args['offset'])
                elif H_args['type'] == "txt":
                    ham_list = Hlib.txt_build(H_args['filename'],H_args['cutoff'],H_args['renorm'],H_args['offset'],H_args['tol'])
                elif H_args['type'] == "list":
                    ham_list = H_args['list']
                if H_args['spin']['bool']:
                    ham_spin_double = Hlib.spin_double(ham_list,len(self.basis))
                    ham_list = ham_list + ham_spin_double   
                    if H_args['spin']['soc']:
                        ham_so = Hlib.SO(self.basis)
                        ham_list = ham_list + ham_so
                    if 'order' in H_args['spin']:
                        if H_args['spin']['order']=='F':
                            ham_FM = Hlib.FM_order(self.basis,H_args['spin']['dS'])
                            ham_list = ham_list + ham_FM
                        elif H_args['spin']['order']=='A':
                            ham_AF = Hlib.AFM_order(self.basis,H_args['spin']['dS'],H_args['spin']['p_up'],H_args['spin']['p_dn'])
                            ham_list = ham_list + ham_AF
                H_obj = gen_H_obj(ham_list)
                return H_obj
            except KeyError:
                print('Invalid dictionary input for Hamiltonian generation.')
                return None
        else:
            return None
        
        
    def solve_H(self):
        '''
        This function diagonalizes the Hamiltonian over an array of momentum vectors.
        It uses the **mat_el** objects to quickly define lambda functions of 
        momentum, which are then filled into the array and diagonalized.
        
        *return*:
            - **self.Eband**: numpy array of float, shape(len(self.Kobj.kpts),len(self.basis)),
            eigenvalues
            
            - **self.Evec**: numpy array of complex float, shape(len(self.Kobj.kpts),len(self.basis),len(self.basis))
            eigenvectors
                
        '''
        if self.Kobj is not None:
            Hmat = np.zeros((len(self.Kobj.kpts),len(self.basis),len(self.basis)),dtype=complex) #initialize the Hamiltonian
            
            for me in self.mat_els:
                Hfunc = me.H2Hk() #transform the array above into a function of k
                Hmat[:,me.i,me.j] = Hfunc(self.Kobj.kpts) #populate the Hij for all k points defined
        
            self.Eband,self.Evec = np.linalg.eigh(Hmat,UPLO='U') #diagonalize--my H_raw definition uses i<=j, so we want to use the upper triangle in diagonalizing
            return self.Eband,self.Evec
        else:
            print('You have not defined a set of kpoints over which to diagonalize.')
            return False
            
        
        
    def plotting(self,win_min=None,win_max=None): #plots the band structure. Takes in Latex-format labels for the symmetry points indicated in the main code
        '''
        Plotting routine for a tight-binding model evaluated over some path in k.
        If the model has not yet been diagonalized, it is done automatically
        before proceeding.
        
        *kwargs*:
            - **win_min**, **win_max**: float, vertical axis limits for plotting
            in units of eV. If not passed, a reasonable choice is made which 
            covers the entire eigenspectrum.
            
        *return*:
            - **ax**: matplotlib axes object
        
        '''
        try:
            Emin,Emax = np.amin(self.Eband),np.amax(self.Eband)
        except AttributeError:
            print('Warning: Bandstructure and energies have not yet been defined. Diagonalizing now.')
            self.solve_H()
            Emin,Emax = np.amin(self.Eband),np.amax(self.Eband)
            print('Diagonalization complete. Proceeding to plotting.')
            
        fig=plt.figure()
        fig.set_tight_layout(False)
        ax=fig.add_subplot(111)
        ax.axhline(y=0,color='k',lw=1.5,ls='--')
        for b in self.Kobj.kcut_brk:
            ax.axvline(x = b,color = 'k',ls='--',lw=1.5)
        for i in range(len(self.basis)):
            ax.plot(self.Kobj.kcut,np.transpose(self.Eband)[i,:],color='navy',lw=1.5)

        plt.xticks(self.Kobj.kcut_brk,self.Kobj.labels)
        if win_max==None or win_min==None:
            ax.set_xlim(self.Kobj.kcut[0],self.Kobj.kcut[-1])
            ax.set_ylim(Emin-1.0,Emax+1.0)
        elif win_max !=None and win_min !=None:
            ax.set_xlim(self.Kobj.kcut[0],self.Kobj.kcut[-1])
            ax.set_ylim(win_min,win_max) 
        ax.set_ylabel("Energy (eV)")

        return ax            
            
def gen_H_obj(htmp):
    '''
    Take a list of Hamiltonian matrix elements in list format:
    [i,j,Rij[0],Rij[1],Rij[2],Hij(R)] and generate a list of **H_me**
    objects instead. This collects all related matrix elements for a given
    orbital-pair for convenient generation of the matrix Hamiltonians over
    an input array of momentum
    
    *args*:
        - **htmp**: list of numeric-type values (mixed integer[:2], float[2:5], complex-float[-1])
    
    *return*:
        - **Hlist**: list of Hamiltonian matrix element, **H_me** objects
    '''

    htmp = sorted(htmp,key=itemgetter(0,1,2,3,4))
    
    Hlist = []
    Hnow = H_me(0,0)
    Rij = np.zeros(3)
    
    for h in htmp:
        if h[0]!=Hnow.i or h[1]!=Hnow.j:
            Hlist.append(Hnow)
            Hnow = H_me(int(np.real(h[0])),int(np.real(h[1])))
        Rij = np.real(h[2:5])
        Hnow.append_H(*Rij,h[5])
    Hlist.append(Hnow)
    return Hlist 
            
