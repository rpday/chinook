# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 08:56:16 2018

@author: rday
"""


import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib


def AFM(TB,D):
    '''
    Give different on-site potential for the different orbitals dxz/dyz and put dxy highest in energy
    '''
    m_copy = TB.mat_els
    
    for ti in range(len(TB.mat_els)):
        if TB.mat_els[ti].i==TB.mat_els[ti].j:
            if TB.mat_els[ti].i<3 or TB.mat_els[ti].i>8:
                m_copy[ti].H.append([0,0,-D])#spin down on A and spin up on B lower in energy
                print(TB.basis[TB.mat_els[ti].i].label,TB.basis[TB.mat_els[ti].i].pos,TB.basis[TB.mat_els[ti].i].spin,-D)
            else:
                m_copy[ti].H.append([0,0,D]) #spin up on A and spin down on B higher in energy
                print(TB.basis[TB.mat_els[ti].i].label,TB.basis[TB.mat_els[ti].i].pos,TB.basis[TB.mat_els[ti].i].spin,D)
    
    return m_copy

def AFO(TB,D):
    '''
    Give different on-site potential to different spin-characters
    '''
    m_copy = TB.mat_els
    for ti in range(len(TB.mat_els)):
        
        if TB.mat_els[ti].i==TB.mat_els[ti].j:
            if np.mod(TB.mat_els[ti].i,6)==0 or np.mod(TB.mat_els[ti].i,6)==4:
                print(TB.basis[TB.mat_els[ti].i].label,TB.basis[TB.mat_els[ti].i].pos,TB.basis[TB.mat_els[ti].i].spin,-D)
                m_copy[ti].H.append([0,0,0,-D]) #dxz site A and dyz site B lower in energy
            elif np.mod(TB.mat_els[ti].i,6)==1 or np.mod(TB.mat_els[ti].i,6)==3:
                m_copy[ti].H.append([0,0,0,D]) #dxz site B and dyz site A higher in energy
                print(TB.basis[TB.mat_els[ti].i].label,TB.basis[TB.mat_els[ti].i].pos,TB.basis[TB.mat_els[ti].i].spin,D)
            elif np.mod(TB.mat_els[ti].i,3)==2:
                m_copy[ti].H.append([0,0,3*D])
            
    return m_copy

if __name__=="__main__":
    a,c =  5.0,5.0
    avec = np.array([[a,0,0],[0,a,0],[0,0,c]])
    
    SK = {"032":0.0,"003322S":1.0,"003322P":1.0,"003322D":0.0}
    CUT,REN,OFF,TOL=a*1.2,1,0.0,0.001
    G,M,X,Z=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([0.5,0,0]),np.array([0,0,0.5])
	

    spin = {'soc':True,'lam':{0:0.02}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0],
			'Z':{0:23},
			'orbs':[["32xz","32yz","32xy"],["32xz","32yz","32xy"]],
			'pos':[np.zeros(3),np.sqrt(0.5)*(avec[0]+avec[1])],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[X,G,M],
			'grain':100,
			'labels':['X','$\Gamma$','M']}


    Hd = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}
 
    	#####
    Bd = build_lib.gen_basis(Bd,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.mat_els = AFO(TB,0.2)
#    TB.mat_els = AFM(TB,0.2)
    TB.solve_H()
    TB.plotting(-1.5,1.5)