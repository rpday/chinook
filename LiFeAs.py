# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:07:08 2018

@author: rday
"""

    
import numpy as np
import matplotlib.pyplot as plt
import build_lib
import ARPES_lib as ARPES
import scipy.ndimage as nd
import operator_library as ops
import plt_sph_harm as sph

if __name__=="__main__":
    a,c =  3.7914,6.3639
    avec = np.array([[a/np.sqrt(2),a/np.sqrt(2),0.0],[-a/np.sqrt(2),a/np.sqrt(2),0.0],[0.0,0.0,c]])

    filenm ="LiFeAs_RD.txt"
    CUT,REN,OFF,TOL=a*3,1,0.061,0.001
    G,X,M,Z,A = np.array([0,0,0]),np.array([0,0.5,0]),np.array([0.5,0.5,0]),np.array([0,0,0.5]),np.array([0.5,0.5,0.5])

	

    
    slab_dict = {'bool':False,
                'hkl':np.array([0,0,1]),
                'cells':20,
                'buff':8,
                'term':0,
                'avec':avec}

    basis_dict = {'atoms':[0,0],
			'Z':{0:26},
			'orbs':[["32xy","32XY","32xz","32yz","32ZR"],["32xy","32XY","32xz","32yz","32ZR"]],
			'pos':[np.array([-a/np.sqrt(8),0,0]),np.array([a/np.sqrt(8),0,0])],
            'slab':slab_dict}

    K_dict = {'type':'F',
			'pts':[M,G,X],
			'grain':1000,
			'labels':['$M$','$\Gamma$','$Z$']}
    
    spin_dict = {'soc':True,'lam':{0:0.04}}

    Ham_dict = {'type':'txt',
			'filename':filenm,
			'cutoff':CUT,
			'renorm':REN,
			'offset':OFF,
			'tol':TOL,
			'so':spin_dict['soc']}
        

 
    	#####
    basis_dict = build_lib.gen_basis(basis_dict,spin_dict)
    Kobj = build_lib.gen_K(K_dict,avec)
    TB = build_lib.gen_TB(basis_dict,Ham_dict,Kobj)
    TB.solve_H()
#    TB.plotting(-0.3,.25)
    O = ops.LdotS(TB,axis='z',vlims=(-0.5,0.5),Elims=(-0.4,0.25))
    ktuple = ((-1.1718,1.1718,800),(-1.1718,1.1718,800),0.0)
    Ef = -0.05
    tol=0.001
    FS = ops.FS(TB,ktuple,Ef,tol)
#    d1d,d1u = {o.index:o.proj for o in TB.basis[:5]},{(o.index-5):o.proj for o in TB.basis[10:15]}
#    d2d,d2u = {(o.index-5):o.proj for o in TB.basis[5:10]},{(o.index-10):o.proj for o in TB.basis[15:]}
#    d1 = {**d1d,**d1u}
#    d2 = {**d2d,**d2u}
#    ind = 8
#    psi_11 = np.array([list(TB.Evec[i,:5,ind])+list(TB.Evec[i,10:15,ind]) for i in range(len(Kobj.kpts))])
#    psi_21 = np.array([list(TB.Evec[i,:5,ind+1])+list(TB.Evec[i,10:15,ind+1]) for i in range(len(Kobj.kpts))])
#    psi_31 = np.array([list(TB.Evec[i,:5,ind+2])+list(TB.Evec[i,10:15,ind+2]) for i in range(len(Kobj.kpts))])
#    psi_41 = np.array([list(TB.Evec[i,:5,ind+3])+list(TB.Evec[i,10:15,ind+3]) for i in range(len(Kobj.kpts))])
#
#    psi_12 = np.array([list(TB.Evec[i,5:10,ind])+list(TB.Evec[i,15:,ind]) for i in range(len(Kobj.kpts))])
#    psi_22 = np.array([list(TB.Evec[i,5:10,ind+1])+list(TB.Evec[i,15:,ind+1]) for i in range(len(Kobj.kpts))])
#    psi_32 = np.array([list(TB.Evec[i,5:10,ind+2])+list(TB.Evec[i,15:,ind+2]) for i in range(len(Kobj.kpts))])
#    psi_42 = np.array([list(TB.Evec[i,5:10,ind+3])+list(TB.Evec[i,15:,ind+3]) for i in range(len(Kobj.kpts))])    
#    
#    psi_1 = np.array([list(TB.Evec[i,:,ind+2]) for i in range(len(Kobj.kpts))])
#    df = {o.index:o.proj for o in TB.basis}
#    
##    
#    for ki in range(len(Kobj.kpts)):
#        psi = psi_1[ki,:]
##        strnm = 'psi_plots\\so_GZ2_4_{:d}.png'.format(ki)
#        sph.gen_psi(30,psi_31[ki],df)#,strnm)
#        
##    ki = 6
#    psi = psi_21[ki,:]
##    strnm = 'psi_plots\\no_so_GZ2_1_{:d}.png'.format(ki)
#    sph.gen_psi(20,psi_11[ki],d1)
#    sph.gen_psi(20,psi_21[ki],d1)
#
#    sph.gen_psi(20,psi_31[ki],d1)
#
#    sph.gen_psi(20,psi_41[ki],d1)

#    sph.gen_psi(20,psi_12[ki],d1)
#    sph.gen_psi(20,psi_22[ki],d1)
#
#    sph.gen_psi(20,psi_32[ki],d1)
#
#    sph.gen_psi(20,psi_42[ki],d1)
    
##    
##    
##    
    ARPES_dict={'cube':{'X':[-0.5,0.5,130],'Y':[-0.5,0.5,130],'kz':0.0,'E':[-0.6,0.1,400]},
                'SE':[0.00,0.005],
                'directory':'C:\\Users\\rday\\Documents\\TB_ARPES\\2018\\TB_ARPES_2018\\LiFeAs',
                'hv': 26.0,
                'pol':np.array([0,1,0]),
                'mfp':7.0,
                'resolution':{'E':0.03,'k':0.05},
                'T':[True,20.0],
                'W':4.0,
                'angle':0.0,
                'spin':None,
                'slice':[False,-0.1]}

##
##    
##
#    exp = ARPES.experiment(TB,ARPES_dict)
#    exp.datacube(ARPES_dict)
#    Ig=exp.plot_gui(ARPES_dict) #If you want to see another polarization, simply update ARPES_dict['pol'] and re-run exp.plot_slice(ARPES_dict)
#

    
    
	#####