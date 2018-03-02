#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:15:00 2018

@author: rday
"""

import numpy as np
import build_lib
import ARPES_lib as ARPES
import klib
import RPA

if __name__=="__main__":
    a,c =  5.839/np.sqrt(2),9.321
    avec = np.array([[a,0,0],[0,a,0],[0.0,0.0,c]])
    
    filenm="Graser.txt"
    CUT,REN,OFF,TOL=10*a,1,0.0,0.001
    G,X,M = np.array([0,0,0]),np.array([0.5,-0.5,0]),np.array([0.5,0.0,0])

    spin = {'soc':False,'lam':{0:0.04}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0],
			'Z':{0:26},
			'orbs':[["32xz","32yz","32XY","32xy","32ZR"]],
			'pos':[np.zeros(3)],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[G,X,M,G],
			'grain':200,
			'labels':['$\Gamma$','X','M','$\Gamma$']}


    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}
 
    	#####
    Bd = build_lib.gen_basis(Bd,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-5,5)
    
    bv = klib.bvectors(avec)
    
    X_1111=RPA.chi_0(TB,[bv[0,0],bv[1,1]],300,0.02,(1,1,1,1))
    
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
##    
#    ARPES_dict={'cube':{'X':[-0.76,0.76,160],'Y':[-0.76,0.76,160],'kz':0.0,'E':[-0.6,0.5,200]},
#                'SE':[0.005,0.00],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                'hv': 21.2,
#                'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
#                'mfp':7.0,
#                'resolution':{'E':0.02,'k':0.01},
#                'T':[True,10.0],
#                'W':4.0,
#                'angle':0.0,
#                'spin':None,
#                'slice':[False,-0.2]}
#
##
##    
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
#    
#    expmt.plot_gui(ARPES_dict)
##
##
