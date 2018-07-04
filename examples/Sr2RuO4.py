#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""

import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_python/TB_ARPES/')

import numpy as np
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES

if __name__=="__main__":

    a,c = 3.903,7.017
    filenm = 'Sr2RuO4_mod.txt'
    c,r,o,t=a*3,1,0,0.001
    G,X,M = np.zeros(3),np.array([0,0.5,0]),np.array([0.5,0.5,0.0])

	
    avec = np.identity(3)*np.array([a,a,c])

    spin = {'soc':True,'lam':{0:0.18}}

    Bd={'atoms':[0],
			'Z':{0:44},
			'orbs':[['42xz','42yz','42xy']],
			'pos':[np.zeros(3)],
            'slab':{'bool':False}}

    Kd = {'type':'F',
			'pts':[X,G,M],
			'grain':200,
			'labels':['X','$\Gamma$','M']}


    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':c,
			'renorm':r,
			'offset':o,
			'tol':t,
			'so':spin['soc']}
    
  	#####



    Bd = build_lib.gen_basis(Bd,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-1.5,1.0)
      
#    
#    ARPES_dict={'cube':{'X':[-0.805,0.805,50],'Y':[-0.805,0.805,50],'kz':0.0,'E':[-1.5,0.2,100]},
#                'SE':[0.002,0.03],
#                'directory':'/save/directory/Sr2RuO4',
#                'hv': 28,
#                'pol':np.array([1,0,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.02,'k':0.02},
#                'T':[True,10.0],
#                'W':4.0,
#                'angle':0.0,
#                'spin':None,
#                'slice':[False,-0.2] 
#                }
#
##
##    
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)    
#    expmt.plot_gui(ARPES_dict)
#    
