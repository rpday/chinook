#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""

import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.operator_library as olib

if __name__=="__main__":

    a,c = 3.903,7.017
    filenm = 'Sr2RuO4_mod.txt'
    c,r,o,t=a*3,1,0,0.001
    G,X,M = np.zeros(3),np.array([0,0.5,0]),np.array([0.5,0.5,0.0])

	
    avec = np.identity(3)*np.array([a,a,c])

    spin = {'bool':True,'soc':True,'lam':{0:0.18}}

    Bd={'atoms':[0],
			'Z':{0:44},
			'orbs':[['42xz','42yz','42xy']],
			'pos':[np.zeros(3)],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[-1*X,G,X],
			'grain':200,
			'labels':['X','$\Gamma$','M']}


    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':c,
			'renorm':r,
			'offset':o,
			'tol':t,
			'spin':spin,
            'avec':avec}
    
  	#####



    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-1.5,1.0)
    O = olib.LdotS(TB,'z',vlims=(-0.5,0.5),Elims=(-1.5,0.1))
      
#    
#    ARPES_dict={'cube':{'X':[-0.805,0.805,150],'Y':[-0.805,0.805,150],'kz':0.0,'E':[-1.5,0.2,200]},
#                'SE':[0.002,0.03],
#                'directory':'/save/directory/Sr2RuO4',
#                'hv': 28,
#                'pol':np.array([1,0,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.02,'k':0.04},
#                'T':[True,10.0],
#                'W':4.0,
#                'angle':np.pi/4,
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
