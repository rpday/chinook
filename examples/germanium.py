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





if __name__=="__main__":
    a =  4.194
    avec = np.array([[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2]])
    
    
    REN,OFF,TOL=1,0.0,0.001
    G,X,W,L,U,K = np.zeros(3),np.array([0,0.5,0.5]),np.array([0.25,0.75,0.5]),np.array([0.5,0.5,0.5]),np.array([0.25,0.625,0.625]),np.array([0.375,0.75,0.375])
	
    fnm = 'C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/examples/germanium_params.txt'
    tol = 0.0001
    SK,CUT = CuTB.gen_Cu_SK(avec,fnm,tol)
#    OFF = CuTB.pair_pot(fnm,avec)/(-10)
#    SK = SK[0]
#    CUT = float(CUT[0])
    
#    SK = {"040":-2.408,"031":4.00,"032":-5.00,"004400S":-0.05652552534871882,"003410S":0.10235422439959821,"004302S":-0.036994370838375354,
#          "003311S": 0.21924434953910618,"003311P":0.0,"003312S":-0.053580360262257626,"003312P":0.013802597521248922,"003322S":-0.012755569410002986,
#          "003322P":0.0033741803350209204,"003322D":-0.0012785070616424799}

    spin = {'bool':False,'soc':True,'lam':{0:0.05}}

    Bd = {'atoms':[0],
			'Z':{0:32},
			'orbs':[["40","41x","41y","41z","50"]],
			'pos':[np.zeros(3)],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[L,G,X,U,G],
			'grain':20,
			'labels':['L','$\Gamma$','X','U','$\Gamma$']}


    Hd = {'type':'SK',
			'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([1,1,1]),
      'thick':6,
      'vac':30,
      'termination':(0,0)}
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-20,20)