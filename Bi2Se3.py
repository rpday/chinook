# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 20:05:23 2018

@author: rday
"""

import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops

if __name__=="__main__":
    a,c =  4.1141,28.64704 
    avec = np.array([[0,a/np.sqrt(3),c/3.],[-a/2.,-a/np.sqrt(12),c/3.],[a/2.,-a/np.sqrt(12),c/3.]])

#    G,Z,F,L = np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0]),np.array([0,0,0.5])

    G,K,M = np.array([0.0,0.0,0.0],float),0.2*np.array([0.0,4*np.pi/3./a,0.0],float),0.2*np.array([2*np.pi/np.sqrt(3)/a,0.0,0.0])

    nu,mu=0.792,0.399
    
    SK1 = {"060":-10.7629,"061":0.2607,"140":-10.9210,"141":-1.5097,"240":-13.1410,"241":-1.1893,\
           "016400S":-0.6770,"016401S":2.0774,"016410S":-0.4792,"016411S":2.0595,"016411P":-0.4432,\
           "026400S":-0.2410,"026401S":-0.2012,"026410S":-0.0193,"026411S":2.0325,"026411P":-0.5320,\
           "114400S":-0.3326,"114401S":-0.0150,"114411S":0.9449,"114411P":-0.1050}

    SK2 = {"006600S":0.2212,"006601S":-0.3067,"006611S":0.3203,"006611P":-0.0510,\
           "114400S":-0.0640,"114401S":0.2833,"114411S":0.3047,"114411P":-0.0035,\
           "224400S":-0.0878,"224401S":-0.2660,"224411S":-0.1486,"224411P":-0.0590}

    SK3 = {"124400S":0.0229,"124401S":-0.0318,"124410S":-0.0778,"124411S":-0.0852,"124411P":0.0120,\
           "006600S":-0.0567,"006601S":-0.2147,"006611S":0.1227,"006611P":-0.0108,\
           "016400S":0.0333,"016401S":-0.0047,"016410S":0.2503,"016411S":-0.1101,"016411P":0.1015} #switched 016410S from 104601S--should be the exact same#also changed 214401S to 124401S
    
    SK_list = [SK1,SK2,SK3]
    CUT = [3.4,4.2,4.75]
    REN,OFF,TOL=1,0.4,0.001


    spin = {'soc':True,'lam':{0:2.066*2./3,1:0.3197*2./3,2:0.3632*2./3}}
    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':4,
                'buff':1,
                'term':2,
                'avec':avec}

    Bd = {'atoms':[1,0,2,0,1],
			'Z':{0:83,1:34,2:34},
			'orbs':[["40","41x","41y","41z"],["60","61x","61y","61z"],["40","41x","41y","41z"],["60","61x","61y","61z"],["40","41x","41y","41z"]],
			'pos':[-nu*(avec[0]+avec[1]+avec[2]),-mu*(avec[0]+avec[1]+avec[2]),np.array([0.0,0.0,0.0]),mu*(avec[0]+avec[1]+avec[2]),nu*(avec[0]+avec[1]+avec[2])] #orbital positions relative to origin
,
            'slab':slab_dict}

    Kd = {'type':'A',
			'pts':[K,G,M],
			'grain':20,
			'labels':['K','$\Gamma$','M']}#,'Z','F','$\Gamma$','L']}


    Hd = {'type':'SK',
          'V':SK_list,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin['soc']}

 
    	#####
    Bd = build_lib.gen_basis(Bd,spin)
    slab = Bd['slab']['obj']
    print('length of slab basis: ',len(slab.slab_base))
    slab.plot_lattice(np.array([b.pos for b in slab.slab_base]))
    Kobj = build_lib.gen_K(Kd,avec)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.solve_H()
    TB.plotting(-1.5,1.5)
#    O = ops.LdotS(TB,axis=None,vlims=(-0.5,0.5),Elims=(-0.5,0.5))
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
##
#    exp = ARPES.experiment(TB,ARPES_dict)
#    exp.datacube(ARPES_dict)
#    exp.plot_slice(ARPES_dict)
#    X,Y,E,I = exp.spectral(ARPES_dict)
=


    

    
    
	#####