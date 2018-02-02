#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import numpy as np
import build_lib
import ARPES_lib as ARPES

if __name__=="__main__":

    a,c = 3.903,7.017
    filenm = 'Sr2RuO4.txt'
    c,r,o,t=a*3,1,0,0.001
    G,X,M = np.zeros(3),np.array([0,0.5,0]),np.array([0.5,0.5,0.0])

	
    avec = np.identity(3)*np.array([a,a,c])

    spin = {'soc':True,'lam':{0:0.11}}

    basis={'atoms':[0],
			'Z':{0:44},
			'orbs':[['42xz','42yz','42xy']],
			'pos':[np.zeros(3)]}

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
    
    x = np.linspace(-np.pi/a,np.pi/a,80)
    X,Y = np.meshgrid(x,x)
    kz = 0.0
    E = np.linspace(-1.2,0.5,100)
    cube = {'X':X,'Y':Y,'kz':kz,'E':E}
    G = [0.1,0.1]
    direct = '/Users/ryanday/Documents/UBC/TB_python/November/v3p0/SRO'
    hv = 21.2
    pol = np.array([1,0,0])
    mfp = 7.0
    dE = 0.03
    dk = 0.01
    T = 10.0

	#####
    basis = build_lib.gen_basis(avec,basis,spin)
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(basis,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-1.2,1.0)

    exp = ARPES.experiment(basis,TB,hv,pol,mfp,dE,dk,T,ang=0.0,W=4.0)
    Iu,Id = exp.datacube(cube,G,spin=None,T_eval = True, directory=direct)
	#####