# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:40:20 2018

@author: rday

Generate a supercell with an integer number of lattice constants. Read in a text-file Hamiltonian, spit out a new orbital basis dictionary and a new text file Hamiltonian
To scale correctly, go sequentially along the axes of the supercell: First create a supercell along a1. Next generate along a2. Finally along a3. In this way can handle
ordering of basis orbitals easily. 

User input:
    basis dictionary
    textfile string


"""


import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.H_library as Hlib

def init_bulk(basis_dict,H_args):
    '''
    Generate bulk model:
    '''
    
    basis_dict = build_lib.gen_basis(basis_dict,{'soc':False})
    Hlist = Hlib.txt_build(*H_args)    
    return basis_dict,Hlist

def gen_spr(basis,H,spr,avec):
    
    '''
    Generate a supercell along indicated direction
    '''
    for s in list(enumerate(spr)):
        H_new = [hi for hi in H] #initialize the Hamiltonian-list
        base_new = [b for b in basis] #
        ncells = s[1]
        tmp_base = []
        for ni in range(1,ncells):
            for o in basis:
                tmp_o = o.copy()
                tmp_o.pos = o.pos + ni*avec[s[0]]
                tmp_base.append(tmp_o)
            H_new += [[hi[0]+ni*len(basis_new),hi[1]+ni*len(basis_new),*hi[2:]] for hi in H_new]
            
        for hij in H_new:
            Rij = np.array([hij[2:5]])
            NR = np.dot(Rij,avec[s[0]])/np.not(avec[s[0]],avec[s[0]])
            
        base_new += tmp_base

            
                


if __name__ == "__main__":
    
    ##lattice vectors
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
    
    ##Basis dictionary
    BD ={'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':{'bool':False}}

    ##H-format
    txt = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/Materials/FeSe_o.txt'
    cutoff,renorm,offset,tol = 10*a,1.0,0.0,1.0e-4
    H_args = [txt,cutoff,renorm,offset,tol]
    
    ## supercell dimensions in units of lattice vectors
    spr = [1,1,0]
    
    ##generate supercell
    BD,H = init_bulk(BD,H_args,spr)