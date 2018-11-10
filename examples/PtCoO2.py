# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:51:03 2018

@author: rday
"""

#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')


import numpy as np
import matplotlib.pyplot as plt
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.ARPES_lib as ARPES
import scipy.ndimage as nd
import ubc_tbarpes.operator_library as ops
from scipy.optimize import curve_fit
import ubc_tbarpes.rotation_lib as rotlib

def trig_onsite(matels,Hco,Hto):
    B = np.array([[1.0j,-np.sqrt(2)*1.0j,0,-np.sqrt(2),-1],[1.0j,-np.sqrt(2)*1.0j,0,np.sqrt(2),1.0],[0,0,np.sqrt(6),0,0],[np.sqrt(2)*1.0j,1.0j,0,1,-np.sqrt(2)],[np.sqrt(2)*1.0j,1.0j,0,-1.0,np.sqrt(2)]])*np.sqrt(1./6)
    T = np.linalg.inv(B).T
    Hc = np.identity(5)*np.array([Hco,Hco,Hto,0,0])
    HcT = np.dot(np.linalg.inv(B),np.dot(Hc,B))
    for ti in matels:
        i,j = np.mod(ti.i,11),np.mod(ti.j,11)
        if i<5 and j<5 and abs(ti.i-ti.j)<5:
            ti.H.append([0,0,0,HcT[i,j]])
    return matels
    



#def _gen_TB():
if __name__ == "__main__":
    a,c =  2.82,17.8 
    avec = np.around(np.array([[0,a,0],[a*np.sqrt(3)/2,a/2,0],[0,0,c]]),5)
    
    RCo,RO1,RO2 = np.zeros(3),a*np.array([-1./np.sqrt(12),0.5,0.33]),a*np.array([np.sqrt(1./12),0.5,-0.33])
    
    ECo = -1.2
    Hco,Hto = 1.0,0.5
    Vpps = 0.5
    Vppp=-0.3
    SK = {'003322S':-0.3,
          '003322P':0.15,
          '003322D':0.0,
          '013221S':-1.6,
          '013221P':0.87,
          '112211S':Vpps,
          '112211P':Vppp,
          '032xy':ECo,
          '032yz':ECo,
          '032ZR':ECo,
          '032xz':ECo,
          '032XY':ECo,
          '121x':-4.0,
          '121y':-4.0,
          '121z':-3.2,#,
          '221z':-7.0,
          '221x':-6.0,
          '221y':-6.0,
          '122211S':Vpps,
          '122211P':Vppp,
          '222211S':Vpps,
          '222211P':Vppp,
          '023221P':0.87,
          '023221S':-1.6}
    
    CUT,REN,OFF,TOL = 3.0,1.0,0.0,0.0001
    
    spin = {'bool':True,'soc':True,'lam':{0:0.07,1:0.0,2:0.0}}
    
    Bd = {'atoms':[0,1,2],
			'Z':{0:27,1:8,2:8},
			'orbs':[["32xy","32yz","32ZR","32xz","32XY"],["21y","21z","21x"],["21y","21z","21x"]],
            'pos':[RCo,RO1,RO2],			
            'spin':spin}
   
    G,K,M = np.array([0,0,0]),np.array([1./3,2./3,0]),np.array([0,0.5,0])

    Kd = {'type':'F',
          'avec':avec,
			'pts':[G,M,K,G],
			'grain':100,
			'labels':['$\Gamma$','M','K','$\Gamma$']}


    Hd = {'type':'SK',
          'V':SK,
          'avec':avec,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'spin':spin}
    
    slab_dict = {'avec':avec,
      'miller':np.array([1,1,1]),
      'thick':10, 
      'vac':30,
      'fine':(0,0),
      'termination':(1,1)}
 

    ARPES_dict={'cube':{'X':[-1.2,1.2,90],'Y':[-1.2,1.2,90],'kz':0.0,'E':[-0.3,0.05,100]},
                'SE':{'cut':27.0,'imfunc':[0.01,0.2]},
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\FeSe',
                'hv': 110.0,
                'pol':np.array([1,0,1]),
                'mfp':7.0,
                'slab':True,
                'resolution':{'E':0.01,'k':0.05},
                'T':[True,10.0],
                'W':4.0,
                'angle':np.pi/2,
                'spin':None,
                'slice':[False,0.0]}



    	#####
    Bd = build_lib.gen_basis(Bd)

    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj)
    TB.mat_els = trig_onsite(TB.mat_els,Hco,Hto)
#    TB.solve_H()
#    TB.plotting(-0.7,0.1)
    
    

         
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.plot_gui(ARPES_dict)
#    return TB

#
#if __name__ == "__main__":
#    
#    TB = _gen_TB()
    
    
    