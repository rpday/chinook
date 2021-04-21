#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Wed Feb 12 09:35:54 2020

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


def importfile(filename):
    
    '''
    Load a wannier90 output file in as a list of numeric
    
    *args*:
        
        - **filename**: string, name of input file
        
    *return*:
        
        - **data**: list of Hamiltonian matrix elements
    
    '''
    data = []
    with open(filename,'r') as fromfile:
        for line in fromfile:
            try:
                li = [float(ll) for ll in line.split()]
                if len(li)==7:
                    data.append([int(li[3])-1,int(li[4])-1,int(li[0]),int(li[1]),int(li[2]),li[5],li[6]])
            except ValueError:
                continue
    return data

def writefile(w90,filename,write_append='w'):
    '''
    Write the parsed wannier90 file to textfile
    
    *args*:
        
        - **w90**: list of numeric, wannier90 Hamiltonian in chinook format
        
        - **filename**: string, name of destination file
        
        - **write_append**: char, 'w' for write, 'a' for append
        
    *return*
    
        - bool
    '''
    
    outfile = filename + '.chinook'
    
    with open(outfile,write_append) as tofile:
        for line in w90:
            formline = [int(line[0]),int(line[1]),*line[2:]]
            tofile.write('{:d},{:d},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f}\n'.format(*formline))
    return True
    
def parse_hr(filename,pos,avec,pos_indices,offset=0,outfile=None):
    '''
    Translate the case_hr.dat file printed by wannier90 into a text-file Hamiltonian
    file which can be read into chinook.
    
    *args*:
        
        - **filename**: string, textfile, wannier hr.dat file
        
        - **pos**: numpy array of Nx3 float, indicating orbital positions
        
        - **avec**: numpy array of 3x3 float, lattice vectors
        
        - **pos_indices**: array of int, relating orbital indices to positions
        
    *kwargs*:
        
        - **offset**: float, offset of chemical potential, eV
        
        - **outfile**: string, filename for output
        
    *return*:
        
        - bool
    
    '''
    
    w90 = np.array(importfile(filename))
    w90[:,2:5] = np.einsum('ji,kj->ki',avec,w90[:,2:5]) 
    w90[:,2:5] += pos[pos_indices[w90[:,0].astype(int)]] - pos[pos_indices[w90[:,1].astype(int)]]
    if outfile is None:
        outfile = filename[:-4]
#    
    writefile(w90,outfile)

    if abs(offset)>0:
        Norbs = w90[:,:2].max()
        onsites = np.array([[ii,ii,0,0,0,offset,0] for ii in range(int(Norbs)+1)])
        writefile(onsites,outfile,write_append='a')
    
    return True


def get_position_pointer(shells):
    '''
    Each of the M atoms in the lattice has some N-number of basis states associated with
    it. We need to relate basis-index from w90 to an atomic site so that we can define the 
    connecting vectors correctly. User passes a list indicating the number of orbitals on
    each atomic site, and we produce the requisite list with this function.

    *args*:

        - **shells**: list of int, length = # atomic sites

    *return*:

        - numpy array of int length = # basis states
    '''

    return np.array([ [jj for jj in range(len(shells)) for ii in range(shells[jj])]]).squeeze()
    



if __name__ == "__main__":
    '''
    I'm going to import a wannier90 HR file here. This is pretty basic, and requires
    some input from you, rather than scrape this data from the w90 input/output files
    directly. You'll need to tell the program what the basis-orbital positions are
    as well as the lattice vectors. In addition, we need to indicate which positions
    correspond to which orbitals -- this is done below using the position pointer: I have 
    2 atomic sites, one has a d-shell (5 orbitals), the other has a p-shell (3 orbitals).
    
    This code will generate a textfile which can be used in a chinook input file by using
    
    hamiltonian_args = {'type':'txt',
        'filename':'my_w90.chinook',
         'cutoff':XXXX,
         'renorm':1.0,
         'offset':0.0,
         'tol':1e-6,
         'avec':avec,
         'spin':spin_args}
    
    in your input file
    
    '''
    
    filename = 'wann_hr.dat'

    avec = np.array([ [  3.0 , 0.0 ,  0.0 ],
                      [  0.0 , 3.0 ,  0.0 ],
                      [  0.0 , 0.0 , 6.0 ]])

    #it's often easier to work with fractional coordinates
    frac_positions = np.array([[0.0, 0.0, 0.2],
                                [0.0, 0.0, 0.6]])#ATOMIC POSITIONS -- LIST OF NP ARRAY -- ANGSTROM
    
    positions = np.dot(frac_positions,avec)

    
    #create a pointer connecting each orbital to an atomic site
    orbital_shells = [5,3] #first site has 5 orbitals (d-shell), second has 3 (p-shell)
    position_pointer = get_position_pointer(orbital_shells)

    

    outfile = 'my_w90'
    
    #execute the transformation
    parse_hr(filename,positions, avec,position_pointer,outfile=outfile)
