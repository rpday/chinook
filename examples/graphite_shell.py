# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:16:28 2018

@author: rday
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:01:41 2018

@author: rday
"""
import sys

sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')



import numpy as np
import ubc_tbarpes.build_lib as build_lib


def build_TB(Kobj):
    a,c =  2.46,3.35
    avec = np.array([[np.sqrt(3)*a/2,a/2,0],
                      [np.sqrt(3)*a/2,-a/2,0],
                      [0,0,2*c]])
    
    filenm = 'graphite.txt'
    CUT,REN,OFF,TOL=a*5,1,0.0,0.001

    spin = {'bool':False,'soc':False,'lam':{0:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0,0,0],
			'Z':{0:6},
			'orbs':[["21z"],["21z"],["21z"],["21z"]],
			'pos':[np.array([0,0,0]),np.array([-a/np.sqrt(3.0),0,0]),np.array([0,0,c]),np.array([-a/(2*np.sqrt(3)),a/2,c])], #OK, consistent with model
            'slab':slab_dict,
            'spin':spin}

    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin,
            'avec':avec,
            'spin':spin}
 
    Bd = build_lib.gen_basis(Bd)

    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    
    return TB

