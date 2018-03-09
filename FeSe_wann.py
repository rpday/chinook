#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import numpy as np
import matplotlib.pyplot as plt
import build_lib
import orbital as olib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import Tk_plot

if __name__=="__main__":
    a,c =  3.7707,5.5210
    avec = np.array([[a,0.0,0.0],[0,a,0.0],[0.0,0.0,c]])

    filenm = 'FeSe_QE.txt'
    CUT,REN,OFF,TOL=a*5,1,0.15,0.001
    G,X,M,Z,mM = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([-0.5,-0.5,0])
    
    

	

    spin = {'soc':False,'lam':{0:0.04,1:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0,1,1],
			'Z':{0:26,1:34},
			'orbs':[["32xy","32xz","32yz","32XY","32ZR"],["32xy","32xz","32yz","32XY","32ZR"],["41x","41y","41z"],["41x","41y","41z"]],
			'pos':[np.dot(np.array([0.25,0.75,0.0]),avec),np.dot(np.array([0.75,0.25,0.0]),avec),np.dot(np.array([0.25,0.25,0.26668]),avec),np.dot(np.array([0.75,0.75,0.73332]),avec)],
            'slab':slab_dict}
    
    phi = np.array([-np.pi/4,-np.pi/4,0,0])
    projs =  [[olib.rotate_util(olib.projdict[o[1:]],phi[a]) for o in Bd['orbs'][a]] for a in range(len(Bd['atoms']))]
    Bd['orient'] = projs
    


    Kd = {'type':'F',
			'pts':[M,G,M],
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
    TB.plotting(-1.5,0.5)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
###    
    ## IRON ATOMS ROTATED BY 45 DEGREES IN THE BASIS HERE!!!!! SO ROTATE THEM, USING PHI ROTATION
    ##THAT'S FOR TOMRROW!
    ARPES_dict={'cube':{'X':[-0.3,0.3,60],'Y':[-0.3,0.3,60],'kz':0.0,'E':[-0.3,0.05,120]},
                'SE':[0.005,0.01],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 21.2,
                'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
                'mfp':7.0,
                'resolution':{'E':0.03,'k':0.05},
                'T':[True,10.0],
                'W':4.0,
                'angle':0.0,
                'spin':None,
                'slice':[False,-0.2]}

#
#    

    expmt = ARPES.experiment(TB,ARPES_dict)
    expmt.datacube(ARPES_dict)
    
    expmt.plot_gui(ARPES_dict)
#
#
#    
    
	#####