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
import scipy.ndimage as nd
import operator_library as ops

if __name__=="__main__":
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm = 'FeSe_RD.txt'
    CUT,REN,OFF,TOL=a*3,1,0.0,0.001
    G,X,M,Z,mM = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([-0.5,-0.5,0])

	

    spin = {'soc':True,'lam':{0:0.04}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[mM,G,M],
			'grain':200,
			'labels':['M','$\Gamma$','M']}


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
#    TB.plotting(-1.25,.75)
    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
##    
#    ARPES_dict={'cube':{'X':[-0.4,0.4,80],'Y':[-0.4,0.4,80],'kz':0.0,'E':[-0.3,0.05,35]},
#                'SE':[0.015,0.01],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                'hv': 21.2,
#                'pol':np.array([0,1,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.05,'k':0.05},
#                'T':[True,10.0],
#                'W':4.0,
#                'angle':0.0,
#                'spin':None}
#
#
#    
#
#    exp = ARPES.experiment(TB,ARPES_dict)
##    X,Y,C,shift = exp.datacube(cube,SE,spin=None,T_eval = True, directory=direct)
##    Au,Ad = exp.datacube(cube,G,spin=None,T_eval = True, directory=direct)
#    exp.datacube(ARPES_dict)
#    I = np.zeros((ARPES_dict['cube']['X'][-1],ARPES_dict['cube']['X'][-1],ARPES_dict['cube']['E'][-1]))
#    for p in range(len(exp.pks)):
#        I[int(np.real(exp.pks[p,0,0])),int(np.real(exp.pks[p,0,1])),int(np.real(exp.pks[p,0,2]))] += abs(exp.pks[p,2,0]+exp.pks[p,2,2])**2 + abs(exp.pks[p,1,0]+exp.pks[p,1,2])**2
#    Ig = nd.gaussian_filter(I,(2,2,2))
#    fig = plt.figure()
#    plt.pcolormesh(exp.X,exp.Y,Ig[:,:,10])
#    x = np.linspace(*xlims)
#    es = np.linspace(*Elims)
    


    

    
    
	#####