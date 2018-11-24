#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 21 12:17:01 2018

@author: bzwartsenberg
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.Tk_plot as Tk_plot


if __name__=="__main__":
    a,c =  4.0,10.0 
    avec = np.array([[a,a,0.0],[a,-a,0.0],[0.0,0.0,c]])

    Vo=1.0 #scale such that total bw of t2gs at X is 0.56 - - 1.014 = 1.574
    SK1 = {"052yz":0.0,
          "052xz":0.0,
          "052xy":0.0,
          "052XY":100.0,
          "052ZR":100.0,
          "152yz":0.0,
          "152xz":0.0,
          "152xy":0.0,
          "152XY":100.0,
          "152ZR":100.0,
          "015522S":1.5*Vo, #this will give zero anyway
          "015522P":-1.0*Vo,
          "015522D":0.25*Vo}
    SK2 = {"005522S":1.5*Vo/a, #this will give zero anyway
          "005522P":-1.0*Vo/a,
          "005522D":0.25*Vo/a,
          "115522S":1.5*Vo/a, #this will give zero anyway
          "115522P":-1.0*Vo/a,
          "115522D":0.25*Vo/a}       
          
    SK_list = [SK1,SK2]
    CUT = [1.001*a,1.001*a*np.sqrt(2)]          
    REN,OFF,TOL=1,0.0,0.001

    G = np.array([0.0,0.0,0.0])
    X = np.array([np.pi/a,0.0,0.0])
    Y = np.array([0.0,np.pi/a,0.0])
    N = np.array([0.5*np.pi/a,0.5*np.pi/a,0.0])	
    M = np.array([0.5*np.pi/a,0.5*np.pi/a,0.0])	
    Z = np.array([0.0,0.0,np.pi/c])

    spin = {'bool':True,'soc':True,'lam':{0:0.5,1:0.5}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    rot =22.0
    
    Bd = {'atoms':[0,1],
			'Z':{0:77,1:77},
			'orbs':[["52yz","52xz","52xy"],["52yz","52xz","52xy"]],
			'pos':[np.zeros(3),np.array([a,0,0])],
                  'orient' : [[rot*np.pi/180],[-rot*np.pi/180]],
            'slab':slab_dict,
            'spin':spin}

    Kd = {'type':'A',
			'pts':[G,M,X,G],
			'grain':100,
			'labels':['$\Gamma$','$X$','$N$','$\Gamma$','$Z$']}


    Hd = {'type':'SK',
          'V':SK_list,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin,
                 'avec':avec}
 
 
    	#####
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(build_lib.gen_basis(Bd),Hd,Kobj)
    TB.solve_H()
    Emin = -3.5
    Emax = 5.5
    vmin = -0.66
    vmax = 1.0
    TB.plotting(Emin,Emax)
    O = ops.LdotS(TB,axis=None,vlims=(vmin,vmax),Elims=(Emin,Emax))
#    plt.savefig('LdotS.pdf',format = 'pdf')
    


