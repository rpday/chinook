#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Sat Nov 18 11:15:35 2017

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

import sys
if sys.version_info<(3,0):
    raise ('This software requires Python 3.0 or higher. Please update your Python installation before proceeding')
else:
    import chinook.orbital as olib
    import chinook.TB_lib as TBlib
    import chinook.slab as slib
    import chinook.klib as klib




###Build Basis
def gen_basis(basis):
    '''
    Generate a list of orbital objects as the input basis for a tight-binding model.
    User passes a basis dictionary, function returns a modified version of this
    same dictionary, with the list of orbitals now appended as the *'bulk'* entry
    
    *args*:

        - **basis**--dictionary with keys:
            
            - *'atoms'*: list of integer, indices for distinct atoms, 
            
            - *'Z'*: dictionary of integer: *'atom'*:element (integer) pairs
            
            - *'orbs'*: list of lists of string, for each atom containing the
            orbital labels (usually in conventional nlxx format)), 
            
            - *'pos'*: list of numpy arrays of length 3 float indicating
            positions of the atoms in direct Angstrom units, 
            
            - optional keys: 
    
                - *'orient'*: list, one entry for each atom, indicating a
                local rotation of the indicated atom, various formats accepted; 
                for more details, c.f. **chinook.orbital.py**
                
                - *'spin'*: dictionary of spin information:
                
                    - *'bool'*: boolean, double basis into spinor basis, 
                    
                    - *'soc'*: boolean, include spin-orbit coupling
                    
                    - *'lam'*: dictionary of SOC constants, integer:float
                    pairs for atoms in *'atoms'* list, and lambda_SOC in eV

                    
    *return*:

        - **basis** dictionary, modified to include the **bulk** list of orbital
        objects

        
    ***
    '''
	
    bulk_basis = []
    
    required = ['atoms','orbs','pos','Z']
    all_present = recur_product([ri in basis.keys() for ri in required])
    do_orient = 'orient' in basis.keys()
    if do_orient:
        for a in range(len(basis['atoms'])):
            if len(basis['orient'][a])==1:
                basis['orient'][a] = [basis['orient'][a] for i in range(len(basis['orbs'][a]))]
            elif len(basis['orient'][a])<len(basis['orbs'][a]):
                raise ValueError ('ORIENT ERROR: pass either 1 orientation per orbital for a given atom, or a single orientation for all orbitals on atom')
                return None
    if not all_present:
        raise ValueError ('BASIS GENERATION ERROR!!!! Ensure atoms, atomic numbers, orbitals, positions are all passed to gen_basis in the basis dictionary. See gen_basis.__doc__ for details.')
        return None
    else:
        for a in list(enumerate(basis['atoms'])):
            for o in list(enumerate(basis['orbs'][a[0]])):
                if do_orient:
            
                    bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]],orient=basis['orient'][a[0]][o[0]]))
                else:
                    bulk_basis.append(olib.orbital(a[1],len(bulk_basis),o[1],basis['pos'][a[0]],basis['Z'][a[1]]))
    if 'spin' in basis.keys():
        if basis['spin']['bool']:
            bulk_basis = olib.spin_double(bulk_basis,basis['spin']['lam'])  
    basis['bulk'] = bulk_basis
    
    return basis

def gen_K(Kdic):    
    '''
    Generate k-path for TB model to be diagonalized along.
    
    *args*: 

        - **Kdic**: dictionary for generation of kpath with:
        
            - *'type'*: string 'A' (absolute) or 'F' (fractional) units 
        
            - *'avec'*: numpy array of 3x3 float lattice vectors
        
            - *'pts'*: list of len3 array indicating the high-symmetry points 
            along the path of interest
        
            - *'grain'*: int, number of points between *each* element of *'pts'*
        
            optional: 
        
                - *'labels'*:list of strings with same length as *'pts'*, giving
                plotting labels for the kpath
            
    *return*:

        **Kobj**: K-object including necessary attributes to be read by the **TB_model**
    
    
    '''
    
    if 'labels' not in Kdic.keys():
        Kdic['labels'] = ['K{:d}'.format(i) for i in range(len(Kdic['pts']))]
    required = ['type','pts','grain']
    if not recur_product([ri in Kdic.keys() for ri in required]):
        raise KeyError('Invalid K-dictionary format. See documentation for gen_K to ensure all required arguments are passed in k-dictionary')
        return None
    if Kdic['type'] == 'F' and 'avec' not in Kdic.keys():
        raise KeyError('Invalid K-dictionary format. Must pass lattice vectors for fractional coordinates')
        return None
    else:
        if Kdic['type']=='F':
            B = klib.bvectors(Kdic['avec'])
            klist = [np.dot(k,B) for k in Kdic['pts']]
        elif Kdic['type']=='A':
            klist = [k for k in Kdic['pts']]
        else:
            klist = []
            print('You have not entered a valid K path. Proceed with caution.')
        Kobj = klib.kpath(klist,Kdic['grain'],Kdic['labels'])
    
        return Kobj


###Built Tight Binding Model
def gen_TB(basis_dict,hamiltonian_dict,Kobj=None,slab_dict=None):
    '''
    Build a Tight-Binding Model using the user-input dictionaries
    
    *args*:

        - **basis_dict**: dictionary, including the *'bulk'* key value pair
        generated by **gen_basis**
        
        - **hamiltonian_dict**: dictionary,
        
            - *'spin'*: same dictionary as passed to **gen_basis** 
            
            - *'type'*: string, Hamiltonian type--'list' (list of matrix elements),
            'SK' (Slater-Koster dictionaries, requires also a 'V' and 'avec' entry),
            'txt' (textfile, requires a 'filename' key as well)
            
            - *'cutoff'*: float, cutoff hopping distance
            
            - *'renorm'*: optional float, renormalization factor default to 1.0
            
            - *'offset'*: optional float, offset of chemical potential, default to 0.0
            
            - *'tol'*: optional float, minimum matrix element tolerance, default to 1e-15
                
        - **Kobj**: optional, standard K-object, as generated by **gen_K**
        
        - **slab_dict**: dictionary for slab generation
        
            - *'avec'*: numpy array of 3x3 float, lattice vectors
            
            - *'miller'*: numpy array of 3 integers, indicating the Miller 
            index of the surface normal in units of lattice vectors
            
            - *'fine'*: fine adjustment of the slab thickness, tuple of two 
            numeric to get desired termination correct (for e.g. inversion symmetry)
            
            - *'thick'*: integer approximate number of unit cells in the
            slab (will not be exact, depending on the fine, and termination
            
            - *'vac'*: int size of the vacuum buffer -- must be larger than
            the largest hopping length to ensure no coupling of slabs
            
            - *'termination'*: tuple of 2 integers: atom indices which 
            terminate the top and bottom of the slab
        

    *return*:

        **TB_model**: tight-binding object, as defined in **chinook.TB_lib.py**
    
    '''
    if 'spin' not in hamiltonian_dict.keys():
        # if omitted, assume no spin-degree of freedom desired
        print('No spin-information entered, assuming no spin-degree of freedom in the following. See build_lib.py for details if spin is desired.')
        hamiltonian_dict['spin']={'bool':False}
    required = ['type','cutoff']
    
    if not recur_product([ri in hamiltonian_dict.keys() for ri in required]):
        raise ValueError ('Ensure all requisite arguments passed in the Hamiltonian dictionary. see gen_TB documentation for details.')
        return None
    else:
        if 'renorm' not in hamiltonian_dict.keys():
            hamiltonian_dict['renorm'] = 1.0
        if 'offset' not in hamiltonian_dict.keys():
            hamiltonian_dict['offset'] = 0.0
        if 'tol' not in hamiltonian_dict.keys():
            hamiltonian_dict['tol'] = 1e-15
        if hamiltonian_dict['type']=='SK' and ('V' not in hamiltonian_dict.keys() or 'avec' not in hamiltonian_dict.keys()):
            raise ValueError ('PLEASE INCLUDE THE DICTIONARY OF Slater-Koster elements as "V" in the Hamiltonian dictionary, and lattice vectors "avec" as numpy array of 3x3 float.')
            return None
        elif hamiltonian_dict['type']=='txt' and 'filename' not in hamiltonian_dict.keys():
            raise ValueError ('No "filename" included in Hamiltonian dictionary keys for text-based Hamiltonian entry.')
            return None
        elif hamiltonian_dict['type']=='list' and 'list' not in hamiltonian_dict.keys():
            raise KeyError ('No "list" included in Hamiltonian dictionary keys for list-based Hamiltonian entry.')
            return None
        else:
            if type(slab_dict)==dict:
                if hamiltonian_dict['spin']['bool']:
                    basis_dict['bulk'] = basis_dict['bulk'][:int(len(basis_dict['bulk'])/2)]
                    Hspin = True
                    hamiltonian_dict['spin']['bool'] = False #temporarily forestall incorporation of spin
                else:
                    Hspin=False
            TB = TBlib.TB_model(basis_dict['bulk'],hamiltonian_dict,Kobj)
            if type(slab_dict)==dict:
                slab_dict['TB'] = TB
                print('running bulk_to_slab now')
                TB,slab_H,Rmat = slib.bulk_to_slab(slab_dict) 
                if Hspin:
                    TB.basis = olib.spin_double(list(TB.basis),basis_dict['spin']['lam']) 
        
                hamiltonian_dict['type']='list'
                hamiltonian_dict['list'] = slab_H
                
                hamiltonian_dict['avec'] = TB.avec
                hamiltonian_dict['spin']['bool']=Hspin
        
                TB.mat_els = TB.build_ham(hamiltonian_dict)
        return TB
    

def recur_product(elements):
    '''
    Utility function: Recursive evaluation of the product of all elements in a list
    
    *args*:

        - **elements**: list of numeric type
    
    *return*:
    
        - product of all elements of **elements**
    
    ***
    '''
    if len(elements)==1:
        return elements[0]
    else:
        return elements[0]*recur_product(elements[1:])



