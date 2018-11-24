# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:47:18 2018

@author: rday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.NBL_TB as CuTB
import ubc_tbarpes.SK as SKlib
import ubc_tbarpes.slab as slib


if __name__=="__main__":
    a =  5.0
    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
  
    CUT,REN,OFF,TOL=3.6,1,0.0,0.001
    G,X,W,L,K = np.zeros(3),np.array([0,0.5,0.5]),np.array([0.25,0.75,0.5]),np.array([0.5,0.5,0.5]),np.array([0.375,0.75,0.375])
	
    SK = {"043":0.0,"004433S":0.1,"004433P":0.2,"004433D":0.3,"004433F":0.4}

    spin = {'bool':False,'soc':True,'lam':{0:1.0}}

    Bd = {'atoms':[0],
			'Z':{0:58},
			'orbs':[['43z3','43xz2','43yz2','43xzy','43zXY','43xXY','43yXY']],
			'pos':[np.zeros(3)],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[G,X,W,L,G,K],
			'grain':100,
			'labels':['$\Gamma$','X','W','L','$\Gamma$','K']}


    Hd = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
#    TB = build_lib.gen_TB(Bd,Hd,Kobj)
#    TB.solve_H()
#    TB.plotting()
    
    
#    ARPES_dict={'cube':{'X':[-0.2,0.2,60],'Y':[-0.2,0.2,60],'kz':0.0,'E':[-0.1,0.05,90]},
#        'SE':[0.002],
#        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
#        'hv': 37,
#        'pol':np.array([0,1,0]),
#        'mfp':7.0,
#        'resolution':{'E':0.005,'k':0.01},
#        'T':[True,10.0],
#        'W':4.0,
#        'angle':0,
#        'spin':None,
#        'slice':[False,-0.005]}
#    
#    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
#    ARPES_expmt.plot_gui(ARPES_dict)