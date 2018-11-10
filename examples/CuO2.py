# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:59:03 2018

@author: rday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.NBL_TB as CuTB
import ubc_tbarpes.SK as SKlib
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.operator_library as ops
import ubc_tbarpes.FS_tetra as FS
import ubc_tbarpes.slab as slib

def SE(w,s,d,p):
    return -1.0j*s+d**2/(w+1.0j*p)


if __name__=="__main__":
    a =  3.80
    c = 6.75
    avec = np.array([[a,0,0],[0,a,0],[0,0,c]])
    t,tp,D = 0.08,-0.4,-0.025
    CUT,REN,OFF,TOL=2.7,1,-0.44,0.001
    SK = {'032':-D,'121':0,
      '003322S':0,
      '003322P':0,
      '003322D':0,
      '112211S':tp,
      '112211P':-tp,
      '013221S':-t*2/np.sqrt(3),
      '013221P':0.0}

    G,X,M,Z = np.zeros(3),np.array([0,0.5,0.0]),np.array([0.5,0.5,0.0]),np.array([0.0,0.0,0.5])



#    SK = CuO_SK(a/2.)

    spin = {'bool':False,'soc':True,'lam':{0:1.1,1:1.0}}

    Bd = {'atoms':[0,1,1],
			'Z':{0:29,1:8},
			'orbs':[["32XY"],["21x"],["21y"]],
			'pos':[np.zeros(3),0.5*avec[0],0.5*avec[1]],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[G,X,M,G,Z],
			'grain':100,
			'labels':['$\Gamma$','X','M','$\Gamma$']}


    Hd = {'type':'SK',
          'V':SK,
#			'filename':'CuO2_3.txt',
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([0,0,1]),
      'thick':20,
      'vac':10,
      'fine':(0,0),
      'termination':(0,0)}
    
    
    
    
    ARPES_dict={'cube':{'X':[0.27,0.5,200],'Y':[-0.0,0.0,1],'kz':0.0,'E':[-0.15,0.1,200]},
            'SE':[0.02,0.01],
            'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
            'hv': 30,
            'pol':np.array([0,1,0]),
            'mfp':7.0,
            'slab':True,
            'resolution':{'E':0.002,'k':0.03},
            'T':[True,150.0],
            'W':4.0,
            'angle':np.pi/4,
            'spin':None,
            'slice':[False,0.0]
           }
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    


    TB.solve_H()
    TB.plotting()

    
    ARPES_dict['SE']=lambda w: SE(w,0.011,0.013,0.006)
    
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.datacube(ARPES_dict)
    I,Ig = ARPES_expmt.spectral(ARPES_dict)    
    
    x = np.linspace(*ARPES_dict['cube']['X'])
    w = np.linspace(*ARPES_dict['cube']['E'])
    W,X = np.meshgrid(w,x)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.pcolormesh(X,W,Ig[0,:,:],cmap=cm.Greys,vmax=0.005)
    ax.set_xlim(0.32,0.475)
    ax.set_ylim(-0.1,0.07)