#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:17:01 2017

@author: ryanday
"""
import sys
sys.path.append('/Users/ryanday/Documents/UBC/TB_python/TB_ARPES-rpday-patch-2/')

import ubc_tbarpes.build_lib as build_lib
import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.plt_sph_harm as sph
import ubc_tbarpes.direct as direct

if __name__=="__main__":
    a,c =  3.7734,5.5258 
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm = 'FeSe_o.txt'
    CUT,REN,OFF,TOL=a*3,1/1.4,0.2,0.001
    G,X,M,Z,mM = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,-0.5,0])

#    pt1,pt2,pt3 = np.zeros(3),np.array([0.2119,0.2119,0]),np.array([0.2119,-0.2119,0])

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
			'pts':[M,G],
			'grain':500,
			'labels':['X','$\Gamma$','X','X','$\Gamma$']}


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
    
    
#    for ii in range(len(TB.mat_els)):
#        if TB.mat_els[ii].i==TB.mat_els[ii].j:
#            if ii<len(TB.basis)/2:
#                TB.mat_els[ii].H.append([0,0,0,1.0e-9])
#            else:
#                TB.mat_els[ii].H.append([0,0,0,-1.0e-9])
    TB.solve_H()
    TB.plotting(-0.2,1)
#    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.25,0.1))
    jdos = direct.path_dir_op(TB,Kobj,0.33,0.1,10)
##    
    Ix = direct.plot_jdos(Kobj,TB,jdos,np.array([1,0,0]))
###    Iy = integral.plot_jdos(Kobj,TB,jdos,np.array([0,1,0]))
###    Iz = integral.plot_jdos(Kobj,TB,jdos,np.array([0,0,1]))
####    
    Ixdos = direct.k_integrated(Kobj,TB,0.02,Ix)
#    
    
#    d1d,d1u = {o.index:o.proj for o in TB.basis[:5]},{(o.index-5):o.proj for o in TB.basis[10:15]}
#    d2d,d2u = {(o.index-5):o.proj for o in TB.basis[5:10]},{(o.index-10):o.proj for o in TB.basis[15:]}
#    d1 = {**d1d,**d1u}
#    d2 = {**d2d,**d2u}
#    ind = 10
##    for oind in range(7,13):
##        ind = oind
#    psi_11 = np.array([list(TB.Evec[i,:5,ind])+list(TB.Evec[i,10:15,ind]) for i in range(len(Kobj.kpts))])
#    
#    psi_12 = np.array([list(TB.Evec[i,5:10,ind])+list(TB.Evec[i,15:,ind]) for i in range(len(Kobj.kpts))])
#       
#    #    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
#    #    df = {o.index:o.proj for o in TB.basis}
#    #    psi_11 = np.array([list(TB.Evec[i,:5,ind:ind+2])+list(TB.Evec[i,10:15,ind:ind+2]) for i in range(len(Kobj.kpts))])
#        
#    for ki in range(len(Kobj.kpts)):
#        tmp =ki
#        strnm = 'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\plots\\FeSe_kz\\no_SO_{:d}_{:0.1f}Z.png'.format(ind,tmp)
#        sph.gen_psi(30,psi_11[ki],d1,strnm)

#    O = ops.fatbs(proj,TB,vlims=(0,1),Elims=(-1,1),degen=True)
    
#    
#    
#    ARPES_dict={'cube':{'X':[-0.2,0.2,50],'Y':[-0.2,0.2,50],'kz':0.0,'E':[-0.45,0.15,100]},
#                'SE':[0.005,0.01],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
#                'hv': 40,
#                'pol':np.array([1,0,0]),
#                'mfp':7.0,
#                'resolution':{'E':0.01,'k':0.02},
#                'T':[False,10.0],
#                'W':4.0,
#                'angle':0,
#                'spin':None,
#                'slice':[False,-0.35]}
#                #'Brads':{'0-3-2-1':100.0,'0-3-2-3':0.0}}
#
##
##    
#
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
##    expmt.plot_slice(ARPES_dict)
#    
#    
#    expmt.plot_gui(ARPES_dict)


    
    
	#####