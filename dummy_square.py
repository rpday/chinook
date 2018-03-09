# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:28:49 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import direct_transition as direct

if __name__=="__main__":
    a,c =  3.0,7.0
    avec = np.array([[a,0.0,0.0],[0.0,a,0.0],[0.0,0.0,c]])
    
    SK = {"021":-1.4,"020":1.4,"002211S":0.3,"002211P":-0.4,"002200S":0.2,"002201S":0.0}
    CUT,REN,OFF,TOL=1.1*a,1,0.0,0.001
    G,M,X=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([0.5,0.0,0.0])
	

    spin = {'soc':False,'lam':{0:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0],
			'Z':{0:6},
			'orbs':[["20","21x","21y","21z"]],
			'pos':[np.zeros(3)],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[X,G,M,X],
			'grain':200,
			'labels':['X','$\Gamma$','M','X']}


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
    TB.solve_H()
    TB.plotting(-5,5)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
###    
    photo_dict={'cube':{'X':[-1.05,1.05,120],'Y':[-1.05,1.05,120],'kz':0.0,'E':[-3,2.5,200]},
                'hv': 2.7,
                'pol':np.array([1,1,1]),
                'T':10.0,
                'Gamma':0.1,
                'TB':TB
                }
#
    direct_trans = direct.direct(photo_dict)
    Mk = direct_trans.resonant_intensity()