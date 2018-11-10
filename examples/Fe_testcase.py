# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:59:03 2018

@author: rday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ubc_tbarpes.build_lib as build_lib

import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.operator_library as ops


if __name__=="__main__":
    
    a =  5.0
    c = 7.5
    avec = np.array([[a,0,0],[0,a,0],[0,0,c]])
    
    
    G,X,M,Z = np.zeros(3),np.array([0,0.5,0.0]),np.array([0.5,0.5,0.0]),np.array([0.0,0.0,0.5])


    CUT = a*1.1
    REN = 1.0
    OFF = 0.0
    TOL = 1e-4
    
    SK = {'041':2.0,
          '004411S':0.2,
          '004411P':-0.6}

    spin = {'bool':False,'soc':True,'lam':{0:0.5},'order':'N','dS':0.0}

    Bd = {'atoms':[0,0],
			'Z':{0:31},
			'orbs':[["41x","41y","41z"],["41x","41y","41z"]],
			'pos':[np.zeros(3),0.5*np.array([a,a,0])],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[G,X,M,G,Z],
			'grain':100,
			'labels':['$\Gamma$','X','M','$\Gamma$']}


    Hd = {'type':'SK',
          'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([0,0,1]),
      'thick':20,
      'vac':10,
      'fine':(0,0),
      'termination':(0,0)}
    
    
    
    
    ARPES_dict={'cube':{'X':[-0.9,0.9,251],'Y':[-0.9,0.9,251],'kz':0.0,'E':[-3,0.1,100]},
            'SE':[0.02,0,0.2],
            'directory':'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/',
            'hv': 30,
            'pol':np.array([0,1,0]),
            'mfp':7.0,
            'slab':False,
            'resolution':{'E':0.03,'k':0.08},
            'T':[True,150.0],
            'W':4.0,
            'angle':0.0,
            'spin':None,
            'slice':[False,0.0]
           }
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    


    TB.solve_H()
    TB.plotting()
    
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.plot_gui(ARPES_dict)
   