#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:27:34 2019

@author: ryanday
"""

import numpy as np

import chinook.build_lib as build_lib
import chinook.ARPES_lib as arpes_lib
import chinook.operator_library as op_lib
import chinook.orbital_plotting as oplot



def construct_tightbinding(pzonly = False):
    '''
    Helper function for building graphene tight-binding model.
    User can specify if they want just the pz-states included, or
    if not, the full C2p3 basis.
    
    *args*:
        
        - **pzonly**: bool, True if wanting only pz-orbitals
        
    *return*:
        
        - **TB**: tight-binding object
        
        - **kpath**: momentum path object, contains points and labels
        for diagonalization
    
    
    '''
    
    
    #### DEFINE LATTICE UNIT CELL#######
    
    alatt = 2.46
    interlayer = 100.0
        
    avec = np.array([[-alatt/2,alatt*np.sqrt(3/4.),0.0],
                    [alatt/2,alatt*np.sqrt(3/4.),0.0],
                    [0.0,0.0,interlayer]]) 
    
    
    ####### DEFINE ORBITAL BASIS ########
    
    spin_args = {'bool':False}
        
    basis_positions = np.array([[0.0,0.0,0.0],
                                [0.0,alatt/np.sqrt(3.0),0.0]])
    
    if pzonly:
        orbitals = ["21z"]
    else:
        orbitals = ["20","21x","21y","21z"]
        
    
    basis_args = {'atoms':[0,0],
                      'Z':{0:6},
                      'orbs':[orbitals,orbitals],
                      'pos':basis_positions,
                      'spin':spin_args}
    
    
    #### DEFINE HAMILTONIAN ####
    
    
    SK = {"020":-8.81,"021":-0.44,       #onsite energies
          "002200S":-5.279,               #nearest-neighbour Vssσ
          "002201S":5.618,                #nearest-neighbour Vspσ
          "002211S":6.05,"002211P":-3.07} #nearest-neighbour Vppσ,Vppπ
    
        
    hamiltonian_args = {'type':'SK',
                    'V':SK,
                    'avec':avec,
                    'cutoff':alatt*0.7,
                    'spin':spin_args}
    
    
    #### DEFINE MOMENTUM PATH ####
    
    G = np.array([0,0,0]) #gamma point
    K = np.array([1./3,2./3,0]) #BZ corner for graphene
    M = np.array([0,0.5,0.0]) #BZ edge centre
    
    momentum_args= {'type':'F',
                'avec':avec,
                'grain':200,
                'pts':[G,K,M,G],
                'labels':['$\\Gamma$','K','M','$\\Gamma$']}
    
    #### CONSTRUCT MODEL ####
    
    basis = build_lib.gen_basis(basis_args)
    kpath = build_lib.gen_K(momentum_args)
    TB = build_lib.gen_TB(basis,hamiltonian_args,kpath)
    
    return TB,kpath
    


def do_fatbands(TB,projections):
    
    '''
    Calculate the orbitally-projected bandstructure, for a series of 
    orbital projections
    
    *args*:
        
        - **TB**: tight-binding object
        
        - **projections**: list of lists of int, e.g. [[0],[1,2],[2,4,6]]
        
    
    '''
    
    for pi in range(len(projections)):
        
        op_lib.fatbs(projections[pi],TB,vlims=(0,0.2),Elims=(TB.Eband.min()*1.1,TB.Eband.max()*1.1))
    
    
    
#def setup_arpes(TB,Kpt,klimit=0.1,Elimits=[-2,0.2],Npoints=100):
    '''
    Initialize an ARPES experiment over a 2D momentum mesh.

    *args*:

        - **TB**: tight-binding object 

        - **Kpt**: iterable, length 3 of float, indicating centre of calculation

        -**klimit**: float, range of momentum in 1/A about centre Kpt

        - **Elimits**: iterable, length 2 of float, indicating energy range of interest, in eV

        - **Npoints**: int, number of k-points along each dimension

    *return*:

        - **experiment**: experiment object, with matrix elements computed
    '''
def setup_arpes(TB,Kpt,klimit=0.1,Elimits=[-2,0.2],Npoints=100):


    arpes_args={'cube':{'X':[Kpt[0]-klimit,Kpt[0]+klimit,Npoints],
                    'Y':[Kpt[1]-klimit,Kpt[1]+klimit,Npoints],
                    'kz':Kpt[2],
                    'E':[Elimits[0],Elimits[1],1000]},
            
                'SE':['poly',0.01,0,0.1],
                'hv': 21.2,
                'pol':np.array([-1,0,1]),
                'resolution':{'E':0.02,'k':0.005},
                'T':4.2}
    
    experiment = arpes_lib.experiment(TB,arpes_args)
    
    experiment.datacube()
    
    return experiment


def plot_wavefunction(TB,band_index,k_index,nangles=20):

    '''
    Plot orbital projection of eigenvector, assumes already diagonalized

    *args*:

        - **TB**: tight-binding object

        - **band_index**: int, index of band zero-based from low-to-high energy

        - **k_index**: int, index of momentum point in the K-path

    *kwargs*:

        - **nangles**: int, optional, number of points in angular mesh for orbital plotting

    '''

    wavefunction = oplot.wavefunction(basis=TB.basis,vector=TB.Evec[k_index,:,band_index])

    _ = wavefunction.triangulate_wavefunction(nangles)

    

#def semenoff_mass(TB,mass):
    
    '''
    Add Semenoff mass to the Hamiltonian
    
    *args*:
        
        - **TB**: tight-binding model
        
        - **mass**: float, mass term
    
    
    '''
    
def semenoff_mass(TB,mass):

    Hnew = [[0,0,0,0,0,mass/2],
            [1,1,0,0,0,-mass/2]]
    
    TB.append_H(Hnew)
    
    
def haldane_mass(TB,mass):
    '''
    Add Haldane terms to the Hamiltonian
    
    *args*:
        
        - **TB**: tight-binding model
        
        - **mass**: float, mass term
    
    '''

    Hnew = []
    vectors = [TB.avec[0],TB.avec[1],TB.avec[1]-TB.avec[0]]
    for ii in range(2):
        for jj in range(3):
            Hnew.append([ii,ii,*vectors[jj],-(2*ii-1)*0.5j*mass])
            Hnew.append([ii,ii,*(-vectors[jj]),(2*ii-1)*0.5j*mass])

    TB.append_H(Hnew)
    

