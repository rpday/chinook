# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:58:12 2018

@author: rday
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES

if __name__=="__main__":
    a,c =  2.0,1.0
    avec = np.array([[a,0,0],[0,a,0],[0,0,c]])

    filenm = 'dummy.txt'
    CUT,REN,OFF,TOL=a*2.1,1,0,0.001
    G,X,M,Z = np.zeros(3),np.array([0,0.5,0]),np.array([0.5,0.5,0.0]),np.array([0,0,0.5])

	

    spin = {'soc':False,'lam':{0:0.05}}

    basis={'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["40"],["40"]],
			'pos':[np.array([0,0,0]),np.array([a/2,0,0])]}

    Kd = {'type':'F',
			'pts':[M,G,Z],
			'grain':200,
			'labels':['M','$\Gamma$','Z']}


    Hd = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}
    
    slabdict = {'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':5,
                'term':0,
                'avec':avec
            }
    
    xlims = [-np.pi/a,np.pi/a,60]
    
    kz = 0.0
    Elims = [-1.0,0.05,40]
    cube = {'X':xlims,'Y':xlims,'kz':kz,'E':Elims}
    G = [0.015,0.01]
    direct = 'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe'
    hv = 21.2
    pol = np.array([0,1,0])
    mfp = 7.0
    dE = 0.05
    dk = 0.05
    T = 10.0

	#####
    basis = build_lib.gen_basis(avec,basis,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(basis,Hd,Kobj,slabdict)
    TB.solve_H()
    TB.plotting(-2,2)
    
    

#    exp = ARPES.experiment(basis,TB,hv,pol,mfp,dE,dk,T,ang=0.0,W=4.0)
#    X,Y,C,shift = exp.datacube(cube,G,spin=None,T_eval = True, directory=direct)
#    Au,Ad = exp.datacube(cube,G,spin=None,T_eval = True, directory=direct)
#    x = np.linspace(*xlims)
#    es = np.linspace(*Elims)
    

    

    
    
	#####