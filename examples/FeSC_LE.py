# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:36:21 2018

@author: rday
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES


#if __name__=="__main__":
    
    
############################ STRUCTURAL PARAMETERS ############################
a=3.7734 #FeSe
c = 5.5258
avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])
Fe1,Fe2 = np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])

G,X,M,Z,R,A = np.array([0,0,0]),np.array([0.5,0.0,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0,0.5]),np.array([0.5,-0.5,0.5])
Mx = np.array([0.5,-0.5,0.0])

############################ HAMILTONIAN PARAMETERS ############################


CUT,REN,TOL=[a*0.8,1.1*a],1.0,1e-7


# good values along GM, but wild and crazy along X!
OFF = -0.32
SK = {"032xz":-2.0,"032yz":-2.0,"032xy":-1.9,
      "003322S":-0.3,"003322P":0.67,"003322D":-0.25}
OFF = 0
re = 0.7
SK1 = {"032xz":-2.0,"032yz":-2.0,"032xy":-1.9,
      "003322S":-0.05,"003322P":0.55,"003322D":0.1}
SK2= {"003322S":re*(-0.05),"003322P":re*0.55,"003322D":re*0.2}
SK = [SK1,SK2]
######################### MODEL GENERATION PARAMETERS ##########################

spin_dict = {'bool':False,
        'soc':True,
        'lam':{0:0.03},
        'order':'N',
        'dS':0.0,
        'p_up':Fe1,
        'p_dn':Fe2}



basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32xz","32yz"],["32xy","32xz","32yz"]],
			'pos':[Fe1,Fe2],
        'spin':spin_dict}

K_dict = {'type':'F',
          'avec':avec,
        #  'pts':[np.array([-0.48,0,0]),np.zeros(3),np.array([0.2,0,0])],#,Z,R,A,Z],
			'pts':[M,G,X],
			'grain':200,
			'labels':['M','$\Gamma$','X']}


ham_dict = {'type':'SK',
			'V':SK,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin_dict,
         'avec':avec}

slab_dict = {'avec':avec,
      'miller':np.array([0,0,1]),
      'fine':(0,0),
      'thick':30,
      'vac':30,
      'termination':(0,0)}

######################### OPTICS EXPERIMENT PARAMETERS #########################


optics_dict = {'hv':0.36,
      'linewidth':0.01,
      'lifetime':0.02,
      'mesh':40,
      'T':10,
      'pol':np.array([1,0,0])}

######################### ARPES EXPERIMENT PARAMETERS #########################


ARPES_dict={'cube':{'X':[-0.7,0.7,100],'Y':[-0.7,0.7,100],'kz':0.0,'E':[-0.25,0.05,100]},
        'SE':[0.01,0.0,0.4],
        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
        'hv': 37,
        'pol':np.array([1,0,0]),
        'mfp':7.0,
        'slab':False,
        'resolution':{'E':0.065,'k':0.01},
        'T':[True,10.0],
        'W':4.0,
        'angle':0.0,
        'spin':None,
        'slice':[False,-0.005]}
 
################################# BUILD MODEL #################################

def build_TB():
    BD = build_lib.gen_basis(basis_dict)
    Kobj = build_lib.gen_K(K_dict)
    TB = build_lib.gen_TB(BD,ham_dict,Kobj)
    return TB

        


if __name__ == "__main__":
    TB = build_TB()
    TB.solve_H()
    TB.plotting(-0.25,0.05)
    TB.plotting()
#    expmt = ARPES.experiment(TB,ARPES_dict)
#    expmt.plot_gui(ARPES_dict)
    #expmt.datacube(ARPES_dict)
    #expmt.plot_slice(ARPES_dict)
    