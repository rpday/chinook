#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:09:34 2018

@author: berend
"""



import numpy as np
import matplotlib.pyplot as plt
import sys
tbpath = '/Users/berend/Documents/Coding/TB_ARPES/'
sys.path.append(tbpath)
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.Tk_plot as Tk_plot
from ubc_tbarpes import rotation_lib
#import ubc_tbarpes.klib as klib
import h5py    

import operator_lib as op
    


    
if __name__=="__main__":
    a,c =  3.878,12.892
    avec = np.array([[a,0.0,0.0],[0.0,a,0.0],[0.0,0.0,c]])
    D1 = -0.07 #note, this would lower the dxy under the dxz? This is from PRB93.085106
    D2 = 2.6
    D3 = 2.0
    D = 2.0 #charge transfer
    DD = 2.0

    #0 is iridium
    #1 is in plane oxygen
    #2 is apical oxygen
    #--> 01 is the same
    #--> 11 is the same
    #--> 02 is the same as 01
    #--> 12 is the same as 11 (could be scaled too though)
    #--> 22 is scaled down 11
    #apical scale factor:
    afac = 1.0
    kzfac = 2.0
    SK = {"052yz":-D1/3,
          "052xz":-D1/3,
          "052xy":2*D1/3,
          "052XY":D2,
          "052ZR":D3,
          "121x":-D, #in plane oxygens
          "121y":-D, 
          "121z":-D,
          "221x":-D,  #apical oxygens
          "221y":-D,
          "221z":-D,
          "005522S":0.0, #No Ir-Ir hopping
          "005522P":0.0,
          "005522D":0.0,
          "015221S":-1.69,
          "015221P":0.78,
          "025221S":-1.69,
          "025221P":0.78,
          "112211S":0.55,
          "112211P":-0.14,          
          "122211S":0.55*afac,
          "122211P":-0.14*afac, 
          "222211S":0.55*kzfac,
          "222211P":-0.14*kzfac, 
          }
          
    CUT = 0.95*a         
    REN,OFF,TOL=1,0.0,0.001


          
    REN,OFF,TOL=1,0.0,0.001

    
    G = np.array([0.0,0.0,0.0])
    X = np.array([np.pi/a,0.0,0.0])
    Y = np.array([0.0,np.pi/a,0.0])
    N = np.array([np.pi/a,np.pi/a,0.0])	
    M = np.array([0.5*np.pi/a,0.5*np.pi/a,0.0])	
    Z = np.array([0.0,0.0,np.pi/c])
    
#    spin = {'bool':True,'soc':True,'lam':{0:0.4,1:0.0,2:0.0},
#            'order' : 'F',
#            'dS' : 1e-9}
    spin = {'bool':True,'soc':True,'lam':{0:0.4,1:0.0,2:0.0}}

           
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    Bd = {   'atoms':[0,1,1,2,2],
			'Z':{0:77,1:8,2:8},
			'orbs':[["52yz","52xz","52xy","52XY","52ZR"],
                        ["21x","21y","21z"],
                        ["21x","21y","21z"],
                        ["21x","21y","21z"],
                        ["21x","21y","21z"]]	,
                'pos':[np.zeros(3), #Ir1
                        np.dot(avec,np.array([0.5,0.0,0.0])),#O1-0
                        np.dot(avec,np.array([0.0,0.5,0.0])), #O1-1
                        np.array([0.0,0.0,2.09]),#O1-2
                        np.array([0.0,0.0,-2.09])],#O1-3                        np.dot(avec,np.array([0.5,0.0,0.5])),#O2-0
            'slab':slab_dict,
            'spin':spin}
     
            
    Kd = {'type':'A',
			'pts':[G,X,N,G,Z],
			'grain':100,
			'labels':['$\Gamma$','$X$','$N$','$\Gamma$','$Z$']}

    Kd2 = {'type':'A',
			'pts':[X,G,Y],
			'grain':100,
			'labels':['$X$','$\Gamma$','$Y$']}

    Kd3 = {'type':'A',
			'pts':[G,X,M,G,Z],
			'grain':100,
			'labels':['$\Gamma$','$X$','$M$','$\Gamma$','$Z$']}

    Hd = {'type':'SK',
          'V':SK,
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

 
    
    TB.Kobj = Kobj
    TB.solve_H()
    Emin = -6.
    Emax = 6.0
    vmin = -0.666
    vmax = 1.0    
    TB.plotting(Emin,Emax)
    Otot = ops.LdotS(TB,axis=None,vlims=(vmin,vmax),Elims=(Emin,Emax))
#    plt.show()
#    Ox = ops.LdotS(TB,axis='x',vlims=(vmin,vmax),Elims=(Emin,Emax))
#    plt.show()
#    Oy = ops.LdotS(TB,axis='y',vlims=(vmin,vmax),Elims=(Emin,Emax))
#    plt.show()
#    Oz = ops.LdotS(TB,axis='z',vlims=(vmin,vmax),Elims=(Emin,Emax))
#    plt.show()
               

