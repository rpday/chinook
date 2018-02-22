# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:07:08 2018

@author: rday
"""

    
import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops

if __name__=="__main__":
    a,c =  3.7914,6.3639
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm = 'LiFeAs.txt'
    CUT,REN,OFF,TOL=a*3,1,0.0,0.001
    G,X,M,Z,mM = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([-0.5,-0.5,0])

	

    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':slab_dict}

    K_dict = {'type':'F',
			'pts':[X,G,M],
			'grain':200,
			'labels':['M','$\Gamma$','M']}
    
    spin_dict = {'soc':True,'lam':{0:0.04}}

    Ham_dict = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin_dict['soc']}
        

 
    	#####
    basis_dict = build_lib.gen_basis(basis_dict,spin_dict)
    Kobj = build_lib.gen_K(K_dict,avec)
    TB = build_lib.gen_TB(basis_dict,Ham_dict,Kobj)
    TB.solve_H()
    TB.plotting(-0.3,.25)
    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.3,0.25))
##    
##    
    ARPES_dict={'cube':{'X':[-0.4,0.4,120],'Y':[-0.4,0.4,120],'kz':0.0,'E':[-0.3,0.05,20]},
                'SE':[0.015,0.01],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 21.2,
                'pol':np.array([0,1,0]),
                'mfp':7.0,
                'resolution':{'E':0.03,'k':0.05},
                'T':[True,10.0],
                'W':4.0,
                'angle':0.0,
                'spin':None,
                'slice':[True,-0.1]}

#
#    
#
    exp = ARPES.experiment(TB,ARPES_dict)
    exp.datacube(ARPES_dict)
    Ig=exp.plot_slice(ARPES_dict) #If you want to see another polarization, simply update ARPES_dict['pol'] and re-run exp.plot_slice(ARPES_dict)


    
    
	#####