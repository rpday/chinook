# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:47:18 2018

@author: rday
"""
import sys
sys.path.append('C:/Users/rday/Documents/TB_ARPES/2018/TB_ARPES_2018/TB_ARPES-master/')

import numpy as np
import ubc_tbarpes.build_lib as build_lib
import ubc_tbarpes.NBL_TB as CuTB
import ubc_tbarpes.ARPES_lib as ARPES
import ubc_tbarpes.operator_library as ops



if __name__=="__main__":
    a =  3.56

    avec = np.around(2.51730014/2.51735729*np.array([[-1.2587, -0.7267,  2.0554],[ 1.2587, -0.7267,  2.0554],[-0.    ,  1.4534,  2.0554]]),4)
    
    CUT,REN,OFF,TOL=3.5,1,0,0.001
    G,X,W,L,K = np.zeros(3),np.array([0,0.5,0.5]),np.array([0.25,0.75,0.5]),np.array([0.5,0.5,0.5]),np.array([0.375,0.75,0.375])
	

    fnm = 'cu_params_pan.txt'
    tol = 0.0001
    SK,CUT = CuTB.gen_Cu_SK(avec,fnm,tol)
    OFF = CuTB.pair_pot(fnm,avec)/(-18)+1.55#-18#

    spin = {'bool':True,'soc':True,'lam':{0:0.06}}

    Bd = {'atoms':[0],
			'Z':{0:29},
			'orbs':[["40","31x","31y","31z","32xy","32xz","32yz","32ZR","32XY"]],
			'pos':[np.zeros(3)],
            'spin':spin}

    Kd = {'type':'F',
          'avec':avec,
			'pts':[G,X,W,L],
			'grain':100,
			'labels':['$\Gamma$','X','W','L','$\Gamma$','K']}


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
      'thick':18,
      'vac':10,
      'fine':(0,0),
      'termination':(0,0)}
    
    Bd = build_lib.gen_basis(Bd)
    Kobj = build_lib.gen_K(Kd)
    TB = build_lib.gen_TB(Bd,Hd,Kobj,slab_dict)
    

    
    
#    G,M,K=np.zeros(3),np.array([0.5,0.5,0.0]),np.array([1./3,2./3,0.0])
#    G,M,K = np.zeros(3),np.array([1.24795,0.72052,0.0]),np.array([0.83197,1.44103,0.0])
#    
#    Kd['type']='A'
#    Kd['avec']=TB.avec
#    Kd['pts'] = [G,M,K,G]
#    TB.Kobj = build_lib.gen_K(Kd)
#    Hd['avec'] = TB.avec

    TB.solve_H()
    TB.plotting()
##    
#    theta =np.linspace(0,2*np.pi,200)
#    kvals = np.array([[0.1*np.cos(t),0.1*np.sin(t),0] for t in theta])
# 
#    
#    
#    TB.Kobj.kpts = kvals
#    TB.Kobj.kcut = theta
#    TB.Kobj.kcut_brk = [0,2*np.pi]
#    TB.Kobj.labels=['0','2pi']
#    TB.solve_H()
###    TB.plotting(-1.5,1.5)
####    
#    Svx = ops.S_vec(len(TB.basis),np.array([1,0,0]))
#    Svy = ops.S_vec(len(TB.basis),np.array([0,1,0]))
#    Sproj = ops.surface_proj(TB.basis,10)
#    Sxs = np.dot(Svx,Sproj)
#    Sys = np.dot(Svy,Sproj)
#    SX = ops.O_path(Sxs,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1,-0.),degen=True)
#    SY = ops.O_path(Sys,TB,Kobj=TB.Kobj,vlims=(-0.25,0.25),Elims=(-1,-0.),degen=True)
#    
    
    ARPES_dict={'cube':{'X':[-0.5,0.5,100],'Y':[-0.5,0.5,100],'kz':0.0,'E':[-0.55,0.05,100]},
        'SE':[0.04,0.0,0.1],
        'directory':'/Users/ryanday/Documents/UBC/TB_ARPES-082018/examples/FeSe',
        'hv': 21.2,
        'pol':np.array([0,1,0]),
        'mfp':5.0,
        'slab':True,
        'resolution':{'E':0.01,'k':0.05},
        'T':[True,10.0],
        'W':4.0,
        'angle':0.0,
        'spin':None,
        'slice':[False,-0.005]}
    
    ARPES_expmt = ARPES.experiment(TB,ARPES_dict)
    ARPES_expmt.plot_gui(ARPES_dict)