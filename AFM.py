#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:01:41 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import Tk_plot

if __name__=="__main__":
    a,c =  5.0,10.0
    avec = np.array([[a,a,0.0],[a,-a,0.0],[0.0,0.0,c]])
    
    to = -1
    SK = {"020":0,"002200S":to}
    CUT,REN,OFF,TOL=a*1.1,1,0.0,0.001
    G2,G,M,X=np.array([1,1,0]),np.zeros(3),np.array([0.5,0.5,0.0]),np.array([0.5,0.0,0.0])
	

    spin = {'soc':False,'lam':{0:0.0,1:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0],
			'Z':{0:3},
			'orbs':[["20"],["20"]],
			'pos':[np.zeros(3),np.array([0.0,a,0.0])],
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
    #ADD AFM ORDER TO HAMILTONIAN
    D = 0.2
#    TB.mat_els[0].H.append([0,0,0,-D])
#    TB.mat_els[3].H.append([0,0,0,D])
#    TB.mat_els[5].H.append([0,0,0,D])
#    TB.mat_els[7].H.append([0,0,0,-D])
    ##FM ORDER
#    TB.mat_els[0].H.append([0,0,0,D])
#    TB.mat_els[3].H.append([0,0,0,D])
#    TB.mat_els[5].H.append([0,0,0,-D])
#    TB.mat_els[7].H.append([0,0,0,-D])
#    
    TB.solve_H()
    TB.plotting(-6,6)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
####    
    ARPES_dict={'cube':{'X':[-0.62,0.62,21],'Y':[-0.62,0.62,21],'kz':0.0,'E':[-5.0,0.6,100]},
                'SE':[0.02,0.00],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 35,
                'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
                'mfp':7.0,
                'resolution':{'E':0.1,'k':0.003},
                'T':[False,10.0],
                'W':4.0,
                'angle':0.0,
                'spin':None,
                'slice':[False,-0.2]}

#
#    

    expmt = ARPES.experiment(TB,ARPES_dict)
    expmt.datacube(ARPES_dict)
#    _,_ = expmt.spectral(ARPES_dict)
    expmt.plot_gui(ARPES_dict)
###
####
