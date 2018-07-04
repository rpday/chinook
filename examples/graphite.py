#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:01:41 2018

@author: rday
"""
import sys

sys.path.append('/Users/ryanday/Documents/UBC/TB_python/TB_ARPES-rpday-patch-2/')



import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.Tk_plot as Tk_plot
import ubc_tbarpes.klib as klib
import ubc_tbarpes.direct as integral

if __name__=="__main__":
    a,c =  2.46,3.35
#    avec = np.array([[np.sqrt(3)*a/2,-a/2,0],[0,-a,0],[0,0,2*c]])
    avec = np.array([[np.sqrt(3)*a/2,a/2,0],[np.sqrt(3)*a/2,-a/2,0],[0,0,2*c]])
#    avec = np.array([[0,2*a,0],[2*a/np.sqrt(12),a,0],[0,0,2*c]])
    
    filenm = 'graphite.txt'
    CUT,REN,OFF,TOL=a*5,1,0.0,0.001
    G,M,K,H,L,A=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([1./3,-1./3,0]),np.array([1./3,-1./3,0.5]),np.array([0.5,0.5,0.5]),np.array([0,0,0.5])
    G2,K2 = np.array([0,0,0.212]),np.array([1./3,-1./3,0.275])
#    G2 = np.array([0,0,3.783])
    spin = {'soc':False,'lam':{0:0.0,1:0.0}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {'atoms':[0,0,0,0],
			'Z':{0:6},
			'orbs':[["21z"],["21z"],["21z"],["21z"]],
			'pos':[np.array([0,0,0]),np.array([-a/np.sqrt(3.0),0,0]),np.array([0,0,c]),np.array([-a/(2*np.sqrt(3)),a/2,c])], #OK, consistent with model
            'slab':slab_dict}
    #2.61 for original calculation 1.55,1.85
    kz = 2.7
    Kd = {'type':'A',
			'pts':[np.array([0,-1.85,kz]),np.array([0,1.85,kz])],#[np.array([0,1.702,2.1]),np.array([0,1.702,3.68]),np.array([0,1.702,4.75])],
			'grain':2000,
			'labels':['15','49.9','84']}


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
#    TB.plotting()
    sig = integral.optical_conductivity(TB,avec,40,300,0.1)
    
#    jdos = integral.path_dir_op(TB,Kobj,1.19,0.2,10)
##    
#    Ix = integral.plot_jdos(Kobj,TB,jdos,np.array([np.cos(np.pi*10/180),0,np.sin(np.pi*10/180)]))
###    Iy = integral.plot_jdos(Kobj,TB,jdos,np.array([0,1,0]))
###    Iz = integral.plot_jdos(Kobj,TB,jdos,np.array([0,0,1]))
####    
#    Ixdos = integral.k_integrated(Kobj,TB,0.02,Ix)
####    
#    ktuple = ((-0.4,0.4,400),(1.3,2.1,400),0.0)#np.pi/c*0.63)
#    Ef = -1.0
#    tol=0.001
#    FS= ops.FS(TB,ktuple,Ef,tol)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
#    
###    
######    
#    hv = 49.9
#    kz_val  =klib.kz_kpt(hv,1.702,4.4,16.4)
#    ARPES_dict = {'cube': {'X':[-2,2,len(Kobj.kpts)],'Y':[-2,2.0,len(Kobj.kpts)],'kz':kz_val,'E':[-2.,1,315]},                
#                'SE':[0.15,0.00],
#                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\graphite',
#                'hv': hv,
#                'pol':np.array([0,0.707,-0.707]),
#                'mfp':7.0,
#                'resolution':{'E':0.03,'k':0.005},
#                'T':[False,10.0],
#                'W':4.4,
#                'angle':0,
#                'spin':None,
#                'slice':[False,-0.2]}
##
###
###    
##
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.datacube(ARPES_dict)
##    Ipes,Ipesg = expmt.spectral(ARPES_dict,(2,50))
#    expmt.plot_gui(ARPES_dict)
#####    
##    
##    
##
########
