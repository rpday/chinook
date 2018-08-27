#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import numpy as np
import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_ARPES-082018/')

import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.optics as optics
import ubc_tbarpes.operator_library as oper


if __name__=="__main__":
    
    
############################ STRUCTURAL PARAMETERS ############################
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
    Fe1,Fe2 = np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])
    G,X,M,Z,R,A = np.array([0,0,0]),np.array([0,-0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])

############################ HAMILTONIAN PARAMETERS ############################

    filenm ='FeSe_o.txt'
    CUT,REN,OFF,TOL=a*3,1/1.4,0.12,0.001

######################### MODEL GENERATION PARAMETERS ##########################

    spin_dict = {'bool':False,
            'soc':False,
            'lam':{0:0.03},
            'order':'N',
            'dS':0.0,
            'p_up':Fe1,
            'p_dn':Fe2}

    basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[Fe1,Fe2],
            'spin':spin_dict}

    K_dict = {'type':'F',
              'avec':avec,
			'pts':[-M,G,M],
			'grain':200,
			'labels':['M','$\Gamma$','M','X','$\Gamma$']}


    ham_dict = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin_dict,
             'avec':avec}
    
    slab_dict = {'avec':avec,
          'miller':np.array([0,0,1]),
          'thick':20,
          'vac':30,
          'termination':(0,0)}
    
######################### OPTICS EXPERIMENT PARAMETERS #########################

    
    optics_dict = {'hv':0.36,
          'linewidth':0.01,
          'lifetime':0.02,
          'mesh':40,
          'T':10,
          'pol':np.array([1,0,0])}
    
######################### ARPES EXPERIMENT PARAMETERS #########################

    
    ARPES_dict={'cube':{'X':[-0.25,0.25,20],'Y':[-0.25,0.25,20],'kz':0.0,'E':[-0.1,0.05,60]},
            'SE':[0.002],
            'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
            'hv': 37,
            'pol':np.array([0,1,0]),
            'mfp':7.0,
            'resolution':{'E':0.005,'k':0.01},
            'T':[True,10.0],
            'W':4.0,
            'angle':0,
            'spin':None,
            'slice':[False,-0.005]}
 
################################# BUILD MODEL #################################


    basis_dict = build_lib.gen_basis(basis_dict)
    Kobj = build_lib.gen_K(K_dict)
    TB = build_lib.gen_TB(basis_dict,ham_dict,Kobj,slab_dict)
    
############################## CHARACTERIZE MODEL ##############################

                
    TB.solve_H()
    TB.plotting(-0.45,0.15)
    sproj = oper.O_path(oper.surface_projection(TB,10),TB,vlims=(0,1),Elims=(-0.5,0.2))
    
################################# EXPERIMENTS #################################

    
#    optics_exp = optics.optical_experiment(TB,optics_dict)
#    optics_exp.integrate_jdos()

    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.plot_gui(ARPES_dict)
