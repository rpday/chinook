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
    a,c =  2.46,10.0
    avec = np.array([[-a/2,a*np.sqrt(3/4.),0.0],[a/2,a*np.sqrt(3/4.),0.0],[0.0,0.0,c]])
    
    to = -3.0
#    SK = [{"021":1.0,"121":-0.0, "002211S":0.0,"002211P":to},{"002211S":0.0,"002211P":-0.2*to},{"002211S":0.0,"002211P":0.025*to}]
    SK = {"020":-8.81,"021":-0.44,"120":-8.81,"121":-0.44,"012200S":-5.729,"012201S":5.618,"012210S":-5.618,"012211S":6.050,"012211P":-3.07} #Sergej Konschuh, Martin Gmitra, Jaroslav Fabian with offset of Dirac point for substrate
#    CUT=[a*0.6,a*1.05,1.3*a]
    CUT,REN,OFF,TOL=a*0.8,1,0.0,0.001
    G,M,K=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([1./3,2./3,0.0])
	

    spin = {'soc':False,'lam':{0:0.0,1:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,1],
			'Z':{0:6,1:6},
			'orbs':[["20","21x","21y","21z"],["20","21x","21y","21z"]],
			'pos':[np.zeros(3),np.array([0.0,a/np.sqrt(3.0),0.0])],
            'slab':slab_dict}

    Kd = {'type':'F',
			'pts':[K,G,M,K],
			'grain':200,
			'labels':['K','$\Gamma$','M','K']}


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
    TB.plotting(-6,3)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
##    
###    
    ARPES_dict={'cube':{'X':[-0.35,0.35,101],'Y':[-0.35,0.35,101],'kz':0.0,'E':[-6.5,-4.5,200]},
                'SE':[0.02,0.00],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 35,
                'pol':np.array([0,np.sqrt(0.5),-np.sqrt(0.5)]),
                'mfp':7.0,
                'resolution':{'E':0.02,'k':0.003},
                'T':[True,10.0],
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
#
#
